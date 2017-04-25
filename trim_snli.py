splits = ['train','test','dev']

for s in splits:
    in_file = open('data/snli/snli_1.0_'+s+'.txt', 'r')
    out_file = open('data/snli/snli_1.0_'+s+'_trim.txt', 'w')
    for line in in_file:
        if not line.startswith("-"):
            line = line.lower()
            out_file.write(line)
    in_file.close()
    out_file.close()