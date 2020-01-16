swords = set()

with open("text.txt") as f:
    for line in f:
        words = list(line)
        for w in words:
            if w.strip():
                swords.add(w.strip())

with open("../tessdata/configs/wordlist", "w") as g:
    g.write("tessedit_char_whitelist " + "".join(swords) + "\n")