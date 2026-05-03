def unicode_converter(text):
    """Converts Devanagari script to ISO 15919 representation."""

    iso_map = {
        # Independent vowels
        0x0905: "a",  0x0906: "ā",  0x0907: "i",  0x0908: "ī",
        0x0909: "u",  0x090A: "ū",  0x090B: "r̥",  0x090F: "ē",
        0x0910: "ai", 0x0913: "ō",  0x0914: "au",
        0x0911: "ô",
        0x0912: "ö",

        # Consonants
        0x0915: "k",   0x0916: "kh",  0x0917: "g",   0x0918: "gh",  0x0919: "ṅ",
        0x091A: "c",   0x091B: "ch",  0x091C: "j",   0x091D: "jh",  0x091E: "ñ",
        0x091F: "ṭ",   0x0920: "ṭh",  0x0921: "ḍ",   0x0922: "ḍh",  0x0923: "ṇ",
        0x0924: "t",   0x0925: "th",  0x0926: "d",   0x0927: "dh",  0x0928: "n",
        0x092A: "p",   0x092B: "ph",  0x092C: "b",   0x092D: "bh",  0x092E: "m",
        0x092F: "y",   0x0930: "r",   0x0932: "l",   0x0935: "v",
        0x0936: "ś",   0x0937: "ṣ",   0x0938: "s",   0x0939: "h",

        # Matras
        0x093E: "ā",  0x093F: "i",  0x0940: "ī",  0x0941: "u",  0x0942: "ū",
        0x0943: "r̥", 0x0947: "ē",  0x0948: "ai", 0x094B: "ō",  0x094C: "au",

        # Modifiers
        0x0901: "m̐",  0x0902: "ṃ",  0x0903: "ḥ",

        # Nukta forms
        0x0958: "q",  0x0959: "k͟h", 0x095A: "ġ",  0x095B: "z",
        0x095C: "ṛ",  0x095D: "ṛh", 0x095E: "f",  0x095F: "ẏ",
    }

    HALANT = 0x094D
    MATRAS = set(range(0x093E, 0x094D))


    NUKTA = 0x093C

    NUKTA_MAP = {
        0x0915: "q",
        0x0916: "k͟h",
        0x0917: "ġ",
        0x091C: "z",
        0x0921: "ṛ",
        0x0922: "ṛh",
        0x092B: "f",
        0x092F: "ẏ",
    }

    def consume_cluster(text, i):
        cluster = ""
        while i < len(text):
            code = ord(text[i])

            if 0x0915 <= code <= 0x0939:
                # Check for nukta (decomposed form)
                if i + 1 < len(text) and ord(text[i + 1]) == NUKTA:
                    cluster += NUKTA_MAP.get(code, iso_map.get(code, ""))
                    i += 2
                else:
                    cluster += iso_map.get(code, "")
                    i += 1

                # Check for halant (cluster continuation)
                if i < len(text) and ord(text[i]) == HALANT:
                    i += 1
                    continue
                else:
                    break
            else:
                break

        return cluster, i

    output = []
    i = 0

    while i < len(text):
        code = ord(text[i])

        # Unknown character → keep as is
        if code not in iso_map and code != HALANT:
            output.append(text[i])
            i += 1
            continue

        # Handle consonant clusters
        if 0x0915 <= code <= 0x0939:
            cluster, i = consume_cluster(text, i)

            next_code = ord(text[i]) if i < len(text) else None

            # Attach matra if present
            if next_code in MATRAS:
                output.append(cluster + iso_map.get(next_code, ""))
                i += 1
            else:
                output.append(cluster + "a")

            continue

        # Handle matras and others
        if code in iso_map:
            output.append(iso_map[code])

        i += 1

    result = "".join(output)

    # Simple schwa deletion (final 'a')
    def schwa_delete(word):
        if word.endswith("a"):
            return word[:-1]
        return word

    words = result.split()
    words = [schwa_delete(w) for w in words]

    return " ".join(words)

