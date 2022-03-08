import sentencepiece as spm

def apply_sp(text_data):
    # モデルの作成
    sp = spm.SentencePieceProcessor()
    sp.Load("sentencepiece.model")

    #load file
    with open("wiki/original/" + text_data, mode='r') as f:
        lines = f.readlines()

    #apply sentencepiece
    tokenized_lines = []
    for l in lines:
        tokenized_sentence = sp.EncodeAsPieces(l)
        tokenized_sentence = ' '.join(tokenized_sentence)
        tokenized_lines.append(tokenized_sentence)

    del lines

    with open("wiki/tokenized/" + text_data, mode='w') as f:
        #for t in tokenized_lines:
        #    f.write(t)
        f.write('\n'.join(tokenized_lines))

def main():
    apply_sp("wiki.train")
    apply_sp("wiki.valid")
    apply_sp("wiki.test")
    
if __name__ == "__main__":
    main()
