# 🔒 Gated models: how to get access

Some models are free but **gated**: their authors want to know who uses them. You ask once, they approve, and from then
on the zoo downloads the model with your personal **Hugging Face token** (a password-like key that only allows
downloading). It takes about 5 minutes, plus waiting for the approval.

| Model in the zoo | Made by | Request access here | Download |
|---|---|---|---|
| `sam3` | Meta | https://huggingface.co/facebook/sam3 | 3.2 GB (once) |

---

## Step 1: Create a free Hugging Face account

Sign up at **https://huggingface.co/join** and confirm your e-mail. (Skip this if you already have an account.)

## Step 2: Request access to the model

1. Log in, then open **https://huggingface.co/facebook/sam3**.
2. At the top of the page there is a short form (name, affiliation, ...). Fill it in, accept the licence and submit.
3. **Wait for approval.** Meta reviews requests by hand, so this can take from minutes to a few days. You get an
   e-mail, and **https://huggingface.co/settings/gated-repos** shows the status (*Pending* → *Accepted*).

You can already do step 3 and 4 while you wait.

## Step 3: Create a token

1. Go to **https://huggingface.co/settings/tokens** and click **Create new token**.
2. Token type: **Fine-grained**. Name: e.g. `insect-model-zoo`.
3. Under **Repositories**, tick only:
   **☑ Read access to contents of all public gated repos you can access**
4. Click **Create token** and **copy it**. It starts with `hf_`, and you only see it once.

(A classic token of type **Read** works too.)

## Step 4: Save the token (not in `main.py`)

In the terminal, with the `.venv` active, run:

```bash
hf auth login
```

Paste the token when it asks (nothing shows while you paste, that is normal) and press Enter; answer `n` to
*"Add token as git credential?"*. The token is saved in your user folder, outside the zoo, and the zoo finds it
automatically. Done.

**Or** put the token in a file called `hf_token.txt` next to `main.py` (just the token, one line). The zoo reads it,
and git ignores that file (see `.gitignore`), so it is never uploaded, also not from forks. **Or** set the environment
variable `HF_TOKEN` (PowerShell `$env:HF_TOKEN="hf_..."`, macOS/Linux `export HF_TOKEN=hf_...`).

> [!WARNING]
> **Keep your token private: do not paste it into `main.py`.** `HF_TOKEN = "..."` at the top of `main.py` still works,
> but git uploads `main.py` with the code (a `.gitignore` cannot hide one line of a file git tracks), so the zoo warns
> you every time. If a token ever leaks, delete it on https://huggingface.co/settings/tokens and create a new one (the
> old one stops working immediately).

## Step 5: Run it

**UI:** `python main.py` → pick **sam3 (gated)** → type what to look for (e.g. `bee`, or `bee, butterfly`) →
**Detect**. The first time, the Detect button shows the 3.2 GB download progress until it is done.

**Command line:**

```bash
python main.py -m sam3 -p "bee" -i images/test_image.jpg
```

To check your setup: `python main.py --check` shows a `GATED:` status next to `sam3`.

SAM 3 is big: with an NVIDIA GPU (6 GB free or more) an image takes a few seconds. It also runs on CPU, but slowly
(about 40 s per image on a fast laptop, 1 to 1.5 minutes on a 4-core laptop CPU).

---

## If it does not work

| The zoo says | What it means | What to do |
|---|---|---|
| `No Hugging Face token found` | no `hf auth login`, no `hf_token.txt`, no `HF_TOKEN` | step 4 |
| `error 401` | Hugging Face does not know this token | copy it again, or create a new one (step 3) |
| `error 403` | the token works, but you have no access yet | wait for approval (step 2, check the *gated-repos* page), or edit the token and tick the gated-repos box (step 3) |
| very slow / out of memory | SAM 3 is a large model | close other programs, or use a GPU; the zoo switches to CPU by itself if the GPU is too small |

Still stuck? Open an issue: https://github.com/InsectAI-COST-Action/insect-model-zoo/issues
