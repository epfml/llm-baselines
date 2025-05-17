import torch
from src.models import utils as model_utils
from src.config import base as config
from src.data.utils import get_tokenizer

def load_model(checkpoint_path, args):
    model = model_utils.get_model(args)
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    model.to(args.device)
    model.eval()
    return model

def generate(model, tokenizer, prompt, max_new_tokens=30, device='cuda'):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0])

if __name__ == "__main__":
    # Load default config
    args = config.get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    tokenizer = get_tokenizer(args)
    
    # Load model
    model = load_model("exps/YOUR_EXP_ID/ckpt.pt", args)

    # Prompt
    prompt = "The square root of 144 is"
    output = generate(model, tokenizer, prompt, max_new_tokens=30, device=args.device)
    print("\n[Prompt]:", prompt)
    print("[Output]:", output)
