import torch

from cs336_basics.Model.softmax import softmax
from cs336_basics.Tokenizer.tokenizer import Tokenizer


def generate_text(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    if temperature <= 0.0 or top_p <= 0 or top_p > 1:
        raise ValueError("错误的温度或top_p参数")
    
    if prompt == "":
            raise ValueError("没有输入prompt")
    
    prompts = tokenizer.encode(prompt)
    
    tokens = torch.tensor(
        prompts,
        dtype=torch.long,
        device=next(model.parameters()).device
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        if len(prompts) >= model.context_length:
            tokens = tokens[..., -model.context_length:]
        
        generated_tokens = []
        for _ in range(max_new_tokens):
            logits = model(tokens)[0, -1, :]
            q = softmax(logits / temperature, -1)

            if top_p != 1:
                q, q_id = torch.sort(
                    q, dim=-1, descending=True
                )
                pd = torch.cumsum(q, dim=-1)
                num_q = torch.searchsorted(
                    pd, 
                    top_p, side='left').item() + 1

                num_q = min(num_q, q.numel())

                id = torch.multinomial(q, num_samples=1)
                token_id = q_id[id].item()
            else:
                token_id = torch.multinomial(q, num_samples=1).item()
                
            if tokenizer.vocab[token_id] == b"<|endoftext|>":
                break

            generated_tokens.append(token_id)

            new_token = torch.tensor(
                [[token_id]],
                dtype=torch.long,
                device=tokens.device
            )
            tokens = torch.cat([tokens, new_token], dim=-1)

            if tokens.shape[-1] >= model.context_length:
                tokens = tokens[..., -model.context_length:]

    return tokenizer.decode(prompts + generated_tokens)