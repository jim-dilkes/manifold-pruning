import json
modules = ["Train", "Validation", "Test"]

for env in modules:
    file_name = env+"_freq_POS.txt"
    with open(env+"/"+file_name, "r", newline="\n") as f:
        filedata = f.read()
    filedata = filedata.replace(" :SPACE", "")
    lines = filedata.split("\n")
    frequency_bin = {}
    for line in lines:
        for word_pos in line.split(" "):
            if word_pos == "":
                continue
            word, pos = word_pos.split(':', maxsplit=1)
            if pos not in frequency_bin.keys():
                frequency_bin[pos] = [word.lower()]
            else:
                frequency_bin[pos].append(word.lower())
    json_dumps = json.dumps(frequency_bin, indent=4)
    with open(env+"/"+env+"_frequency_bin.json", "w") as f:
        f.write(json_dumps)
