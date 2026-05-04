import re

POORNA_VIRAM = '\u0964'
DBL_DANDA    = '\u0965'

text = "Hello world! This is a test. Another sentence."
parts = re.split(r'([.?!]+)', text)
sents = []
for i in range(0, len(parts)-1, 2):
    sents.append((parts[i] + parts[i+1]).strip())
if len(parts) % 2 == 1 and parts[-1].strip():
    sents.append(parts[-1].strip())
print(sents)
