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

## Step 4: Paste the token into `main.py`

Open `main.py` (e.g. in VS Code). Near the top you find:

```python
HF_TOKEN = ""
```

Paste your token between the quotes and save:

```python
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> [!WARNING]
> **Keep your token private.** Do not share `main.py` with your token in it, and never push it to GitHub. If it
> ever leaks, delete it on https://huggingface.co/settings/tokens and create a new one (the old one stops working
> immediately).

<details>
<summary>Other ways (advanced)</summary>

Instead of editing `main.py` you can:
- set the environment variable `HF_TOKEN` (e.g. PowerShell `$env:HF_TOKEN="hf_..."`, macOS/Linux
  `export HF_TOKEN=hf_...`), or
- log in once with `hf auth login`. The zoo finds the saved login automatically.

</details>

## Step 5: Run it

**UI:** `python main.py` → pick **sam3 (gated)** → type what to look for (e.g. `bee`, or `bee, butterfly`) →
**Detect**. The first time, the 3.2 GB download shows its progress in the result panel.

**Command line:**

```bash
python main.py -m sam3 -p "bee" -i images/test_image.jpg
```

To check your setup: `python main.py --check` shows a `GATED:` status next to `sam3`.

SAM 3 is big: with an NVIDIA GPU (6 GB free or more) an image takes a few seconds. It also runs on CPU, but slowly
(about 40 s per image on a fast laptop).

---

## If it does not work

| The zoo says | What it means | What to do |
|---|---|---|
| `No Hugging Face token found` | `HF_TOKEN` in `main.py` is empty | step 4 |
| `error 401` | Hugging Face does not know this token | copy it again, or create a new one (step 3) |
| `error 403` | the token works, but you have no access yet | wait for approval (step 2, check the *gated-repos* page), or edit the token and tick the gated-repos box (step 3) |
| very slow / out of memory | SAM 3 is a large model | close other programs, or use a GPU; the zoo switches to CPU by itself if the GPU is too small |

Still stuck? Open an issue: https://github.com/HugoMarkoff/Insect_model_zoo/issues
