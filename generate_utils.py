import torch
import time
import math

def vanilla_generate(target, input_ids, input_len, max_gen_len=64, eos_id=2, pad_token=2):
    assert input_ids != None, "please give the input"
    bsz = input_ids.size(0)
    output_ids = input_ids.new_zeros((bsz, max_gen_len))
    
    target.set_max_gen_len(max_gen_len)
    
    cache_lens = input_ids.new_zeros((bsz)).int()
    hidden_states = target.model.forward(input_ids, exec_type="prefill").last_hidden_state
    output_ids[:, 0] = target.lm_head(hidden_states[range(bsz), input_len-1, :]).argmax(dim=-1)
    cache_lens += input_len
    # autoregressive decoding
    for _ in range(1, max_gen_len):
        input_ids = output_ids[range(bsz), cache_lens - input_len].view(bsz, -1)
        hidden_states = target.forward(input_ids, cache_lens=cache_lens.clone(), exec_type="decoding").last_hidden_state
        llm_output = target.lm_head(hidden_states[:, -1, :]).argmax(dim=-1)
        cache_lens += 1
        output_ids[range(bsz), cache_lens - input_len] = llm_output.view(-1)
        if (llm_output.eq(eos_id)).any():
            break
    return output_ids

# def ucb_selection(warmup_step=1, exploration_weight=6.0):
    # if 

def double_buffer_spec_generate(draft, target, input_ids, input_len, gamma=4, max_gen_len=64, eos_id=2, pad_token=2):
    assert input_ids != None, "please give the input"
    bsz = input_ids.size(0)
    output_ids = input_ids.new_zeros((bsz, max_gen_len + gamma)).fill_(pad_token)
    spec_mask = input_ids.new_zeros((bsz, max_gen_len + gamma))
    
    draft.set_max_gen_len(max_gen_len + 128)
    target.set_max_gen_len(max_gen_len + 128)
    
    cache_lens = input_ids.new_zeros((bsz)).int()
    pesudo_hidden_states = target.forward(input_ids, exec_type="prefill")["last_hidden_state"]
    output_ids[:, 0] = target.lm_head(pesudo_hidden_states[range(bsz), input_len-1, :]).argmax(dim=-1)
    spec_mask[:, 0] = 0
    cache_lens += input_len
    draft_cache_lens = cache_lens.clone() - 1

    # Eagle prefill
    hidden_states = torch.cat([draft.embed_tokens(input_ids[:, 1:]), pesudo_hidden_states[:, :-1]], dim=-1)
    position_ids = torch.arange(0, hidden_states.size(1))[None, :].to(input_ids.device)
    position_embeddings = target.model.rotary_emb(hidden_states, position_ids)
    draft_hidden_states = draft.eagle_forward(
        hidden_states=hidden_states, 
        position_embeddings=position_embeddings, 
        cache_lens=draft_cache_lens.clone(), 
        exec_type="prefill",
    )

    # spec tokens
    spec_buffer = output_ids.new_zeros((bsz, gamma + 1))
    spec_buffer[:, 0] = output_ids[:, 0]
    next_spec_start_token = spec_buffer[:, :].clone()[:, 0, None]
    pesudo_hidden_states = pesudo_hidden_states[:, -next_spec_start_token.size(1):, :]
    count = 0
    num = 0

    torch.cuda.synchronize()
    start_time = time.time()
    correct_len = output_ids.new_zeros((bsz)).fill_(1)

    draft_time = 0
    for out_index in range(1, max_gen_len):
        torch.cuda.synchronize()
        draft_start = time.time()
        # speculative decoding
        for spec_steps in range(0, gamma):
            
            if spec_steps == 0:
                word_embed = draft.embed_tokens(next_spec_start_token)
                hidden_states = torch.cat([word_embed, pesudo_hidden_states[:, :next_spec_start_token.size(1)]], dim=-1)
                position_ids = torch.arange(0, next_spec_start_token.size(-1))[None, :].to(input_ids.device) + draft_cache_lens[:, None]
                position_embeddings = target.model.rotary_emb(word_embed, position_ids)
                # print(f"draft model input token {next_spec_start_token[:, :]}, pos id is {position_ids[:, :]}, cut off len is {correct_len - 1}")
            else:
                pesudo_hidden_states = draft_hidden_states[:, -1, None]
                word_embed = draft.embed_tokens(spec_buffer[:, spec_steps, None])
                hidden_states = torch.cat([word_embed, pesudo_hidden_states], dim=-1)# bsz * 1 * 2d
                position_embeddings = target.model.rotary_emb(word_embed, draft_cache_lens[:, None])

            draft_hidden_states = draft.eagle_forward(hidden_states=hidden_states, position_embeddings=position_embeddings, 
                                        cache_lens=draft_cache_lens.clone(), exec_type="decoding")
            
            if spec_steps == 0:
                draft_cache_lens += correct_len
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, :, :]).argmax(dim=-1)[range(bsz), correct_len - 1]
            else:
                draft_cache_lens += 1
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, -1, :]).argmax(dim=-1).view(-1,)
        torch.cuda.synchronize()
        draft_end = time.time()
        draft_time += (draft_end - draft_start)
        draft_cache_lens = draft_cache_lens - gamma + 1
        pesudo_hidden_states = target.forward(spec_buffer, cache_lens=cache_lens.clone(), exec_type="decoding").last_hidden_state
        llm_verify_output = target.lm_head(pesudo_hidden_states[:, -gamma - 1:, :]).argmax(dim=-1)
        # print(f"input to LLM is {spec_buffer}, LLM output is {llm_verify_output}")
        verification = llm_verify_output[:, :-1].eq(spec_buffer[:, 1:]).cumprod(dim=-1)
        correct_len = verification.sum(dim=-1) + 1 # bonus token
        llm_verify_output[:, 1:] = llm_verify_output[:, 1:] * verification

        row_indices = torch.arange(bsz, device=cache_lens.device).unsqueeze(1)
        col_indices = (cache_lens - input_len).unsqueeze(1) + torch.arange(1, gamma + 2, device=cache_lens.device)
        
        output_ids[row_indices, col_indices] = llm_verify_output[:, :gamma + 1]
        bonus_token = llm_verify_output[row_indices.view(-1), correct_len-1]
        # print(f"LLM bonus token is {bonus_token}, pos id is {cache_lens + correct_len - 1}")
        spec_buffer[:, 0] = bonus_token
        # for i in range(bsz):
        #     spec_mask[i, cache_lens[i] - input_len[i] + 1 : cache_lens[i] - input_len[i] + correct_len[i]] = 1
        

        next_spec_start_token = llm_verify_output[:, :correct_len.max()]
        # print(correct_len)

        cache_lens += correct_len
        # print(f"draft cache lens is {draft_cache_lens}, cache lens is {cache_lens}")
        if (cache_lens - input_len).max() + gamma + 2 > output_ids.size(1):
            break
        if (output_ids.eq(eos_id)).any():
            break
        count += (correct_len - 1).sum()
        num += bsz

    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(count, num, count / num + 1)
    return output_ids, count, num, elapsed_time, draft_time


def ucb_length_spec_generate(draft, target, input_ids, input_len, gamma=4, max_gen_len=64, eos_id=2, pad_token=2):
    gamma = 4
    assert input_ids != None, "please give the input"
    bsz = input_ids.size(0)
    output_ids = input_ids.new_zeros((bsz, max_gen_len + gamma)).fill_(pad_token)
    spec_mask = input_ids.new_zeros((bsz, max_gen_len + gamma))
    
    draft.set_max_gen_len(max_gen_len + 128)
    target.set_max_gen_len(max_gen_len + 128)
    
    cache_lens = input_ids.new_zeros((bsz)).int()
    pesudo_hidden_states = target.forward(input_ids, exec_type="prefill")["last_hidden_state"]
    output_ids[:, 0] = target.lm_head(pesudo_hidden_states[range(bsz), input_len-1, :]).argmax(dim=-1)
    spec_mask[:, 0] = 0
    cache_lens += input_len
    draft_cache_lens = cache_lens.clone() - 1

    # Eagle prefill
    hidden_states = torch.cat([draft.embed_tokens(input_ids[:, 1:]), pesudo_hidden_states[:, :-1]], dim=-1)
    position_ids = torch.arange(0, hidden_states.size(1))[None, :].to(input_ids.device)
    position_embeddings = target.model.rotary_emb(hidden_states, position_ids)
    draft_hidden_states = draft.eagle_forward(
        hidden_states=hidden_states, 
        position_embeddings=position_embeddings, 
        cache_lens=draft_cache_lens.clone(), 
        exec_type="prefill",
    )

    # spec tokens
    spec_buffer = output_ids.new_zeros((bsz, gamma + 1))
    spec_buffer[:, 0] = output_ids[:, 0]
    next_spec_start_token = spec_buffer[:, :].clone()[:, 0, None]
    pesudo_hidden_states = pesudo_hidden_states[:, -next_spec_start_token.size(1):, :]
    count = 0
    num = 0

    torch.cuda.synchronize()
    start_time = time.time()
    correct_len = output_ids.new_zeros((bsz)).fill_(1)

    draft_time = 0

    ucb_mean = correct_len.new_zeros((4)).float().fill_(0)
    ucb_calltime = correct_len.new_zeros((4)).fill_(0)
    exploration_weight = 6.0

    for out_index in range(1, max_gen_len):
        torch.cuda.synchronize()
        bandit_start = time.time()
        if out_index < 4 * 2:
            gamma = (out_index - 1) % 4 + 1
        else:
            ucb_score = ucb_mean + math.sqrt(exploration_weight * math.log(out_index) / ucb_calltime.sum())
            gamma = ucb_score.argmax().item() + 1
        # speculative decoding
        for spec_steps in range(0, gamma):
            
            if spec_steps == 0:
                word_embed = draft.embed_tokens(next_spec_start_token)
                hidden_states = torch.cat([word_embed, pesudo_hidden_states[:, :next_spec_start_token.size(1)]], dim=-1)
                position_ids = torch.arange(0, next_spec_start_token.size(-1))[None, :].to(input_ids.device) + draft_cache_lens[:, None]
                position_embeddings = target.model.rotary_emb(word_embed, position_ids)
                # print(f"draft model input token {next_spec_start_token[:, :]}, pos id is {position_ids[:, :]}, cut off len is {correct_len - 1}")
            else:
                pesudo_hidden_states = draft_hidden_states[:, -1, None]
                word_embed = draft.embed_tokens(spec_buffer[:, spec_steps, None])
                hidden_states = torch.cat([word_embed, pesudo_hidden_states], dim=-1)# bsz * 1 * 2d
                position_embeddings = target.model.rotary_emb(word_embed, draft_cache_lens[:, None])

            draft_hidden_states = draft.eagle_forward(hidden_states=hidden_states, position_embeddings=position_embeddings, 
                                        cache_lens=draft_cache_lens.clone(), exec_type="decoding")
            
            if spec_steps == 0:
                draft_cache_lens += correct_len
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, :, :]).argmax(dim=-1)[range(bsz), correct_len - 1]
            else:
                draft_cache_lens += 1
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, -1, :]).argmax(dim=-1).view(-1,)

        draft_cache_lens = draft_cache_lens - gamma + 1
        pesudo_hidden_states = target.forward(spec_buffer[:, :gamma+1], cache_lens=cache_lens.clone(), exec_type="decoding").last_hidden_state
        llm_verify_output = target.lm_head(pesudo_hidden_states[:, -gamma - 1:, :]).argmax(dim=-1)
        assert gamma >= 1, f"{gamma} is invalid"
        verification = llm_verify_output[:, :-1].eq(spec_buffer[:, 1:gamma+1]).cumprod(dim=-1)
        correct_len = verification.sum(dim=-1) + 1 # bonus token
        llm_verify_output[:, 1:] = llm_verify_output[:, 1:] * verification
        row_indices = torch.arange(bsz, device=cache_lens.device).unsqueeze(1)
        col_indices = (cache_lens - input_len).unsqueeze(1) + torch.arange(1, gamma + 2, device=cache_lens.device)
        if col_indices.max().item() + 2 > output_ids.size(1):
            break
        output_ids[row_indices, col_indices] = llm_verify_output[:, :gamma + 1]
        bonus_token = llm_verify_output[row_indices.view(-1), correct_len-1]
        spec_buffer[:, 0] = bonus_token
        if correct_len.max().item() > llm_verify_output.size(1):
            next_spec_start_token = llm_verify_output[:, :correct_len.max().item() - 1]
        else:
            next_spec_start_token = llm_verify_output[:, :correct_len.max().item()]
        cache_lens += correct_len


        if (cache_lens - input_len).min() + gamma + 2 > output_ids.size(1):
            break
        if (output_ids.eq(eos_id)).any():
            break
        count += (correct_len - 1).sum()
        num += bsz
        torch.cuda.synchronize()
        bandit_end = time.time()
        bandit_tokens = correct_len.sum().item()
        bandit_put = bandit_tokens / (bandit_end - bandit_start)
        ucb_mean[gamma -1] = (ucb_mean[gamma - 1] * ucb_calltime[gamma - 1] + bandit_put) / (1 + ucb_calltime[gamma - 1]) 
        ucb_calltime[gamma - 1] += 1
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"{bsz}\tucb_calltime:{ucb_calltime}")
    # print(count, num, count / num + 1)
    return output_ids, count, num, elapsed_time, 0


def ucb_spec_generate_with_batch_mask(draft, target, input_ids, input_len, max_gen_len=64, eos_id=2, pad_token=2, gamma=1, all_quota=156):

    if gamma != -1:
        #if offer gamma, then go to vanilla speculative decoding
        gamma = gamma
        bandit = False
        spec_quota = 0
    else:
        gamma = min(max(all_quota // input_ids.size(0) - 1, 0), 4)
        bandit = True
        spec_quota = min(max( (all_quota - (gamma + 1) * input_ids.size(0) ) // input_ids.size(0), 0), input_ids.size(0))

    assert input_ids != None, "please give the input"

    bsz = input_ids.size(0)
    output_ids = input_ids.new_zeros((bsz, max_gen_len + gamma)).fill_(pad_token)
    spec_mask = input_ids.new_zeros((bsz, max_gen_len + gamma))
    
    draft.set_max_gen_len(max_gen_len + 128)
    target.set_max_gen_len(max_gen_len + 128)
    
    cache_lens = input_ids.new_zeros((bsz)).int()
    pesudo_hidden_states = target.forward(input_ids, exec_type="prefill")["last_hidden_state"]
    output_ids[:, 0] = target.lm_head(pesudo_hidden_states[range(bsz), input_len-1, :]).argmax(dim=-1)
    spec_mask[:, 0] = 0
    cache_lens += input_len
    draft_cache_lens = cache_lens.clone() - 1

    # Eagle prefill
    hidden_states = torch.cat([draft.embed_tokens(input_ids[:, 1:]), pesudo_hidden_states[:, :-1]], dim=-1)
    position_ids = torch.arange(0, hidden_states.size(1))[None, :].to(input_ids.device)
    position_embeddings = target.model.rotary_emb(hidden_states, position_ids)
    draft_hidden_states = draft.eagle_forward(
        hidden_states=hidden_states, 
        position_embeddings=position_embeddings, 
        cache_lens=draft_cache_lens.clone(), 
        exec_type="prefill",
    )

    # spec tokens
    spec_buffer = output_ids.new_zeros((bsz, gamma + 1))
    spec_buffer[:, 0] = output_ids[:, 0]
    next_spec_start_token = spec_buffer[:, :].clone()[:, 0, None]
    pesudo_hidden_states = pesudo_hidden_states[:, -next_spec_start_token.size(1):, :]
    count = 0
    num = 0

    torch.cuda.synchronize()
    start_time = time.time()
    correct_len = output_ids.new_zeros((bsz)).fill_(1)
    # record_time = time.time()
    # autoregressive decoding
    # print(f"draft cache lens is {draft_cache_lens}, cache lens is {cache_lens}")

    expand_llm_output = output_ids.new_zeros((bsz, gamma + 1)).fill_(-1)
    expand_pesudo_hidden_state = pesudo_hidden_states.new_zeros((bsz, gamma + 1, pesudo_hidden_states.size(-1)))
    
    torch.cuda.synchronize()
    start_time = time.time()

    ucb_mean = correct_len.new_zeros((bsz)).float().fill_(0)
    ucb_calltime = correct_len.new_zeros((bsz)).fill_(0)
    ucb_selection = correct_len.new_zeros((bsz,))
    exploration_weight = 6.0
    range_index_flatten = torch.arange(bsz, device=cache_lens.device)

    draft_time = 0.0
    
    for out_index in range(1, max_gen_len):
        # speculative decoding
        torch.cuda.synchronize()
        draft_start = time.time()
        for spec_steps in range(0, gamma):
            
            if spec_steps == 0:
                word_embed = draft.embed_tokens(next_spec_start_token)
                hidden_states = torch.cat([word_embed, pesudo_hidden_states[:, :next_spec_start_token.size(1)]], dim=-1)
                position_ids = torch.arange(0, next_spec_start_token.size(-1))[None, :].to(input_ids.device) + draft_cache_lens[:, None]
                position_embeddings = target.model.rotary_emb(word_embed, position_ids)
                # print(f"draft model input token {next_spec_start_token[:, :]}, pos id is {position_ids[:, :]}, cut off len is {correct_len - 1}")
            else:
                pesudo_hidden_states = draft_hidden_states[:, -1, None]
                word_embed = draft.embed_tokens(spec_buffer[:, spec_steps, None])
                hidden_states = torch.cat([word_embed, pesudo_hidden_states], dim=-1)# bsz * 1 * 2d
                position_embeddings = target.model.rotary_emb(word_embed, draft_cache_lens[:, None])

            draft_hidden_states = draft.eagle_forward(hidden_states=hidden_states, position_embeddings=position_embeddings, 
                                        cache_lens=draft_cache_lens.clone(), exec_type="decoding")
            
            if spec_steps == 0:
                draft_cache_lens += correct_len
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, :, :]).argmax(dim=-1)[range(bsz), correct_len - 1]
            else:
                draft_cache_lens += 1
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, -1, :]).argmax(dim=-1).view(-1,)

        draft_cache_lens = draft_cache_lens - gamma + 1
        # torch.cuda.synchronize()
        # draft_end = time.time()
        # draft_time += (draft_end - draft_start)
        batch_id_mask = spec_buffer.new_zeros(spec_buffer.size()).bool()
        batch_id_mask[:, :gamma+1] = True
        batch_id_mask[:, 0] = True
        ucb_score = ucb_mean + math.sqrt(exploration_weight * math.log(out_index) / ucb_calltime.sum())
        _, index = ucb_score.topk(largest=True, sorted=True, k=spec_quota)
        ucb_selection.fill_(0)
        ucb_selection.scatter_(0, index, 1)

        if (spec_quota != 0) & (bandit):
            batch_id_mask[ucb_selection * range_index_flatten, gamma] = True

        filter_spec_buffer = spec_buffer[batch_id_mask]
        pesudo_hidden_states = target.forward(filter_spec_buffer, cache_lens=cache_lens.clone(), batch_id_mask=batch_id_mask, exec_type="decoding").last_hidden_state
        llm_verify_output = target.lm_head(pesudo_hidden_states).argmax(dim=-1)
        expand_llm_output = spec_buffer.new_zeros(spec_buffer.size()).fill_(0)
        expand_llm_output[batch_id_mask] = llm_verify_output
        llm_verify_output = expand_llm_output
        expand_pesudo_hidden_state[batch_id_mask] = pesudo_hidden_states
        pesudo_hidden_states = expand_pesudo_hidden_state


        verification = llm_verify_output[:, :-1].eq(spec_buffer[:, 1:]).cumprod(dim=-1)
        cutoff = batch_id_mask[:, 1:].ne(0).sum(dim=-1)
        correct_len = verification.sum(dim=-1).clamp(max=cutoff) + 1 # bonus token
        llm_verify_output[:, 1:] = llm_verify_output[:, 1:] * verification

        row_indices = torch.arange(bsz, device=cache_lens.device).unsqueeze(1)
        col_indices = (cache_lens - input_len).unsqueeze(1) + torch.arange(1, gamma + 2, device=cache_lens.device)
        
        output_ids[row_indices, col_indices] = llm_verify_output[:, :gamma + 1]
        bonus_token = llm_verify_output[row_indices.view(-1), correct_len-1]
        spec_buffer[:, 0] = bonus_token

        next_spec_start_token = llm_verify_output[:, :correct_len.max()]
        cache_lens += correct_len

        ucb_mean[ucb_selection * range_index_flatten] = ( ucb_mean[ucb_selection * range_index_flatten] * ucb_calltime[ucb_selection * range_index_flatten] + correct_len[ucb_selection * range_index_flatten]) / (ucb_calltime[ucb_selection * range_index_flatten] + 1)
        ucb_calltime += ucb_selection * 1

        # print(f"draft cache lens is {draft_cache_lens}, cache lens is {cache_lens}")
        if (cache_lens - input_len).max() + gamma + 2 > output_ids.size(1):
            break
        if (output_ids.eq(eos_id)).any():
            break
        count += (correct_len - 1).sum()
        num += bsz
        
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return output_ids, count, num, elapsed_time, draft_time



def double_buffer_spec_generate_with_batch_mask(draft, target, input_ids, input_len, gamma=4, max_gen_len=64, eos_id=2, pad_token=2):
    assert input_ids != None, "please give the input"
    bsz = input_ids.size(0)
    output_ids = input_ids.new_zeros((bsz, max_gen_len + gamma)).fill_(pad_token)
    spec_mask = input_ids.new_zeros((bsz, max_gen_len + gamma))
    
    draft.set_max_gen_len(max_gen_len + 128)
    target.set_max_gen_len(max_gen_len + 128)
    
    cache_lens = input_ids.new_zeros((bsz)).int()
    pesudo_hidden_states = target.forward(input_ids, exec_type="prefill")["last_hidden_state"]
    output_ids[:, 0] = target.lm_head(pesudo_hidden_states[range(bsz), input_len-1, :]).argmax(dim=-1)
    spec_mask[:, 0] = 0
    cache_lens += input_len
    draft_cache_lens = cache_lens.clone() - 1

    # Eagle prefill
    hidden_states = torch.cat([draft.embed_tokens(input_ids[:, 1:]), pesudo_hidden_states[:, :-1]], dim=-1)
    position_ids = torch.arange(0, hidden_states.size(1))[None, :].to(input_ids.device)
    position_embeddings = target.model.rotary_emb(hidden_states, position_ids)
    draft_hidden_states = draft.eagle_forward(
        hidden_states=hidden_states, 
        position_embeddings=position_embeddings, 
        cache_lens=draft_cache_lens.clone(), 
        exec_type="prefill",
    )

    # spec tokens
    spec_buffer = output_ids.new_zeros((bsz, gamma + 1))
    spec_buffer[:, 0] = output_ids[:, 0]
    next_spec_start_token = spec_buffer[:, :].clone()[:, 0, None]
    pesudo_hidden_states = pesudo_hidden_states[:, -next_spec_start_token.size(1):, :]
    count = 0
    num = 0

    torch.cuda.synchronize()
    start_time = time.time()
    correct_len = output_ids.new_zeros((bsz)).fill_(1)
    # record_time = time.time()
    # autoregressive decoding
    # print(f"draft cache lens is {draft_cache_lens}, cache lens is {cache_lens}")

    expand_llm_output = output_ids.new_zeros((bsz, gamma + 1)).fill_(-1)
    expand_pesudo_hidden_state = pesudo_hidden_states.new_zeros((bsz, gamma + 1, pesudo_hidden_states.size(-1)))
    
    draft_time = 0

    torch.cuda.synchronize()
    start_time = time.time()

    for out_index in range(1, max_gen_len):
        # speculative decoding
        for spec_steps in range(0, gamma):
            if spec_steps == 0:
                word_embed = draft.embed_tokens(next_spec_start_token)
                hidden_states = torch.cat([word_embed, pesudo_hidden_states[:, :next_spec_start_token.size(1)]], dim=-1)
                position_ids = torch.arange(0, next_spec_start_token.size(-1))[None, :].to(input_ids.device) + draft_cache_lens[:, None]
                position_embeddings = target.model.rotary_emb(word_embed, position_ids)
                # print(f"draft model input token {next_spec_start_token[:, :]}, pos id is {position_ids[:, :]}, cut off len is {correct_len - 1}")
            else:
                pesudo_hidden_states = draft_hidden_states[:, -1, None]
                word_embed = draft.embed_tokens(spec_buffer[:, spec_steps, None])
                hidden_states = torch.cat([word_embed, pesudo_hidden_states], dim=-1)# bsz * 1 * 2d
                position_embeddings = target.model.rotary_emb(word_embed, draft_cache_lens[:, None])

            draft_hidden_states = draft.eagle_forward(hidden_states=hidden_states, position_embeddings=position_embeddings, 
                                        cache_lens=draft_cache_lens.clone(), exec_type="decoding")
            
            if spec_steps == 0:
                draft_cache_lens += correct_len
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, :, :]).argmax(dim=-1)[range(bsz), correct_len - 1]
            else:
                draft_cache_lens += 1
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, -1, :]).argmax(dim=-1).view(-1,)

        draft_cache_lens = draft_cache_lens - gamma + 1

        batch_id_mask = spec_buffer.new_zeros(spec_buffer.size()).float().normal_().abs().cumsum(dim=-1).le(0.4 * gamma + 0.4)
        batch_id_mask[:, 0] = True
        batch_id_mask[:, :] = True

        filter_spec_buffer = spec_buffer[batch_id_mask]
        pesudo_hidden_states = target.forward(filter_spec_buffer, cache_lens=cache_lens.clone(), batch_id_mask=batch_id_mask, exec_type="decoding").last_hidden_state
        llm_verify_output = target.lm_head(pesudo_hidden_states).argmax(dim=-1)
        expand_llm_output = spec_buffer.new_zeros(spec_buffer.size()).fill_(0)
        expand_llm_output[batch_id_mask] = llm_verify_output
        llm_verify_output = expand_llm_output
        expand_pesudo_hidden_state[batch_id_mask] = pesudo_hidden_states
        pesudo_hidden_states = expand_pesudo_hidden_state


        verification = llm_verify_output[:, :-1].eq(spec_buffer[:, 1:]).cumprod(dim=-1)
        cutoff = batch_id_mask[:, 1:].ne(0).sum(dim=-1)
        correct_len = verification.sum(dim=-1).clamp(max=cutoff) + 1 # bonus token
        llm_verify_output[:, 1:] = llm_verify_output[:, 1:] * verification

        row_indices = torch.arange(bsz, device=cache_lens.device).unsqueeze(1)
        col_indices = (cache_lens - input_len).unsqueeze(1) + torch.arange(1, gamma + 2, device=cache_lens.device)
        
        output_ids[row_indices, col_indices] = llm_verify_output[:, :gamma + 1]
        bonus_token = llm_verify_output[row_indices.view(-1), correct_len-1]
        spec_buffer[:, 0] = bonus_token

        torch.cuda.synchronize()
        draft_end = time.time()

        next_spec_start_token = llm_verify_output[:, :correct_len.max()]

        cache_lens += correct_len
        # print(f"draft cache lens is {draft_cache_lens}, cache lens is {cache_lens}")
        if (cache_lens - input_len).max() + gamma + 2 > output_ids.size(1):
            break
        if (output_ids.eq(eos_id)).any():
            break
        count += (correct_len - 1).sum()
        num += bsz
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(num)
    print(count)
    # print(count, num, count / num + 1)
    return output_ids, count, num, elapsed_time, draft_time

def ucb_v2_spec_generate_with_batch_mask(draft, target, input_ids, input_len, max_gen_len=64, eos_id=2, pad_token=2, spec_quota=0, gamma=1):
    if spec_quota == 0:
        gamma = 0
        cgamma = 0

    else:
        gamma = gamma
        if gamma != 1:
            cgamma = gamma
        else:
            cgamma = gamma
    assert input_ids != None, "please give the input"
    bsz = input_ids.size(0)
    output_ids = input_ids.new_zeros((bsz, max_gen_len + gamma)).fill_(pad_token)
    spec_mask = input_ids.new_zeros((bsz, max_gen_len + gamma))
    
    draft.set_max_gen_len(max_gen_len + 128)
    target.set_max_gen_len(max_gen_len + 128)
    
    cache_lens = input_ids.new_zeros((bsz)).int()
    pesudo_hidden_states = target.forward(input_ids, exec_type="prefill")["last_hidden_state"]
    output_ids[:, 0] = target.lm_head(pesudo_hidden_states[range(bsz), input_len-1, :]).argmax(dim=-1)
    spec_mask[:, 0] = 0
    cache_lens += input_len
    draft_cache_lens = cache_lens.clone() - 1

    # Eagle prefill
    hidden_states = torch.cat([draft.embed_tokens(input_ids[:, 1:]), pesudo_hidden_states[:, :-1]], dim=-1)
    position_ids = torch.arange(0, hidden_states.size(1))[None, :].to(input_ids.device)
    position_embeddings = target.model.rotary_emb(hidden_states, position_ids)
    draft_hidden_states = draft.eagle_forward(
        hidden_states=hidden_states, 
        position_embeddings=position_embeddings, 
        cache_lens=draft_cache_lens.clone(), 
        exec_type="prefill",
    )

    # spec tokens
    spec_buffer = output_ids.new_zeros((bsz, gamma + 1))
    spec_buffer[:, 0] = output_ids[:, 0]
    next_spec_start_token = spec_buffer[:, :].clone()[:, 0, None]
    pesudo_hidden_states = pesudo_hidden_states[:, -next_spec_start_token.size(1):, :]
    count = 0
    num = 0

    torch.cuda.synchronize()
    start_time = time.time()
    correct_len = output_ids.new_zeros((bsz)).fill_(1)
    # record_time = time.time()
    # autoregressive decoding
    # print(f"draft cache lens is {draft_cache_lens}, cache lens is {cache_lens}")

    expand_llm_output = output_ids.new_zeros((bsz, gamma + 1)).fill_(-1)
    expand_pesudo_hidden_state = pesudo_hidden_states.new_zeros((bsz, gamma + 1, pesudo_hidden_states.size(-1)))
    
    torch.cuda.synchronize()
    start_time = time.time()

    ucb_mean = correct_len.new_zeros((bsz)).float().fill_(0)
    ucb_calltime = correct_len.new_zeros((bsz)).fill_(0)
    ucb_selection = correct_len.new_zeros((bsz,))
    exploration_weight = 6.0
    range_index_flatten = torch.arange(bsz, device=cache_lens.device)

    draft_time = 0.0
    
    for out_index in range(1, max_gen_len):
        # speculative decoding
        torch.cuda.synchronize()
        draft_start = time.time()
        for spec_steps in range(0, cgamma):


            
            if spec_steps == 0:
                word_embed = draft.embed_tokens(next_spec_start_token)
                hidden_states = torch.cat([word_embed, pesudo_hidden_states[:, :next_spec_start_token.size(1)]], dim=-1)
                position_ids = torch.arange(0, next_spec_start_token.size(-1))[None, :].to(input_ids.device) + draft_cache_lens[:, None]
                position_embeddings = target.model.rotary_emb(word_embed, position_ids)
                # print(f"draft model input token {next_spec_start_token[:, :]}, pos id is {position_ids[:, :]}, cut off len is {correct_len - 1}")
            else:
                pesudo_hidden_states = draft_hidden_states[:, -1, None]
                word_embed = draft.embed_tokens(spec_buffer[:, spec_steps, None])
                hidden_states = torch.cat([word_embed, pesudo_hidden_states], dim=-1)# bsz * 1 * 2d
                position_embeddings = target.model.rotary_emb(word_embed, draft_cache_lens[:, None])

            draft_hidden_states = draft.eagle_forward(hidden_states=hidden_states, position_embeddings=position_embeddings, 
                                        cache_lens=draft_cache_lens.clone(), exec_type="decoding")
            
            if spec_steps == 0:
                draft_cache_lens += correct_len
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, :, :]).argmax(dim=-1)[range(bsz), correct_len - 1]
            else:
                draft_cache_lens += 1
                spec_buffer[:, spec_steps + 1] = target.lm_head(draft_hidden_states[:, -1, :]).argmax(dim=-1).view(-1,)

        draft_cache_lens = draft_cache_lens - gamma + 1
        torch.cuda.synchronize()
        draft_end = time.time()
        draft_time += (draft_end - draft_start)
        batch_id_mask = spec_buffer.new_zeros(spec_buffer.size()).bool()
        batch_id_mask[:, :gamma] = True
        batch_id_mask[:, 0] = True
        ucb_score = ucb_mean + math.sqrt(exploration_weight * math.log(out_index) / ucb_calltime.sum())
        _, index = ucb_score.topk(largest=True, sorted=True, k=spec_quota)
        ucb_selection.fill_(0)
        ucb_selection.scatter_(0, index, 1)
        if spec_quota != 0:
            batch_id_mask[ucb_selection * range_index_flatten, gamma] = True

        filter_spec_buffer = spec_buffer[batch_id_mask]
        pesudo_hidden_states = target.forward(filter_spec_buffer, cache_lens=cache_lens.clone(), batch_id_mask=batch_id_mask, exec_type="decoding").last_hidden_state
        llm_verify_output = target.lm_head(pesudo_hidden_states).argmax(dim=-1)
        expand_llm_output = spec_buffer.new_zeros(spec_buffer.size()).fill_(0)
        expand_llm_output[batch_id_mask] = llm_verify_output
        llm_verify_output = expand_llm_output
        expand_pesudo_hidden_state[batch_id_mask] = pesudo_hidden_states
        pesudo_hidden_states = expand_pesudo_hidden_state


        verification = llm_verify_output[:, :-1].eq(spec_buffer[:, 1:]).cumprod(dim=-1)
        cutoff = batch_id_mask[:, 1:].ne(0).sum(dim=-1)
        correct_len = verification.sum(dim=-1).clamp(max=cutoff) + 1 # bonus token
        llm_verify_output[:, 1:] = llm_verify_output[:, 1:] * verification

        row_indices = torch.arange(bsz, device=cache_lens.device).unsqueeze(1)
        col_indices = (cache_lens - input_len).unsqueeze(1) + torch.arange(1, gamma + 2, device=cache_lens.device)
        
        output_ids[row_indices, col_indices] = llm_verify_output[:, :gamma + 1]
        bonus_token = llm_verify_output[row_indices.view(-1), correct_len-1]
        spec_buffer[:, 0] = bonus_token

        next_spec_start_token = llm_verify_output[:, :correct_len.max()]
        cache_lens += correct_len

        ucb_mean[ucb_selection * range_index_flatten] = ( ucb_mean[ucb_selection * range_index_flatten] * ucb_calltime[ucb_selection * range_index_flatten] + correct_len[ucb_selection * range_index_flatten]) / (ucb_calltime[ucb_selection * range_index_flatten] + 1)
        ucb_calltime += ucb_selection * 1

        # print(f"draft cache lens is {draft_cache_lens}, cache lens is {cache_lens}")
        if (cache_lens - input_len).max() + gamma + 2 > output_ids.size(1):
            break
        if (output_ids.eq(eos_id)).any():
            break
        count += (correct_len - 1).sum()
        num += bsz
        
    print(input_ids.size(), out_index)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"spec accept len: {count / (out_index * spec_quota) + 1}")
    return output_ids, count, num, elapsed_time, draft_time



def benchmark_speed(target, input_len=128, bsz=1, gamma=4):
    input_ids = torch.randint(0, 10000, (bsz, input_len)).to(target.device)
    bsz = input_ids.size(0)
    output_ids = input_ids
    spec_buffer = input_ids.new_zeros((bsz, gamma + 1))

    cache_lens = input_ids.new_zeros(bsz).fill_(input_len).int()

    warmup_step = 10

    pesudo_hidden_states = target.model.forward(input_ids, exec_type="prefill").last_hidden_state
    for i in range(warmup_step):
        target.forward(input_ids, cache_lens=cache_lens.clone(), exec_type="decoding")


    torch.cuda.synchronize()
    start_time = time.time()

    num_runs = 256
    batch_id_mask = spec_buffer.new_zeros(spec_buffer.size()).float().normal_().abs().cumsum(dim=-1).le(0.4 * gamma + 0.4)
    batch_id_mask[:, :] = True
    batch_id_mask[:, gamma+1:] = False# True
    batch_id_mask[:, 0] = True
    mean_gamma = batch_id_mask.sum() / bsz - 1

    for i in range(num_runs):

        expand_llm_output = output_ids.new_zeros((bsz, gamma + 1)).fill_(-1)
        expand_pesudo_hidden_state = pesudo_hidden_states.new_zeros((bsz, gamma + 1, pesudo_hidden_states.size(-1)))



        filter_spec_buffer = spec_buffer[batch_id_mask]
        pesudo_hidden_states = target.forward(filter_spec_buffer, cache_lens=cache_lens.clone(), batch_id_mask=batch_id_mask, exec_type="decoding").last_hidden_state
        llm_verify_output = target.lm_head(pesudo_hidden_states).argmax(dim=-1)
        expand_llm_output = spec_buffer.new_zeros(spec_buffer.size()).fill_(0)
        expand_llm_output[batch_id_mask] = llm_verify_output
        llm_verify_output = expand_llm_output
        expand_pesudo_hidden_state[batch_id_mask] = pesudo_hidden_states
        pesudo_hidden_states = expand_pesudo_hidden_state

    torch.cuda.synchronize()
    end_time = time.time()

    print(f"input_len-{input_len} bsz-{bsz} mean_gamma-{mean_gamma} time: {(end_time - start_time)/ num_runs} turn/s")