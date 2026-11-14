"""Membuat gambar target dari prompt rahasia dengan seed tetap, lalu menyimpannya ke static/target.png.

Karena target dibuat oleh model gambar yang sama (Z-Image-Turbo) dengan seed yang sama dengan aplikasi, ambang kemiripan pasti bisa
dicapai (prompt rahasia menghasilkan skor 1,0). Prompt rahasia tidak boleh diberi tahu ke partisipan.

Pemakaian:
  python -m scripts.make_target "a red fox sitting in a snowy forest, golden sunrise light"
  python -m scripts.make_target               # memakai TARGET_PROMPT atau data/secret/target_prompt.txt
"""
import sys

from app import config, image_gen


def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]).strip()
        config.TARGET_PROMPT_FILE.write_text(prompt, encoding="utf-8")
    else:
        prompt = config.read_target_prompt()
    img = image_gen.generate(prompt)
    img.save(config.TARGET_IMAGE)
    print(f"Prompt rahasia : {prompt}")
    print(f"Seed           : {config.SEED}")
    print(f"Disimpan di    : {config.TARGET_IMAGE}")


if __name__ == "__main__":
    main()
