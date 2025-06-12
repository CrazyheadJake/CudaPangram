def main():
    with open("words.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    
    wordSets = set()
    words = []
    for line in lines:
        print(line)
        chars = frozenset(line)
        if (chars not in wordSets):
            wordSets.add(chars)
            words.append(line)
    
    with open("include/filteredWords.inl", "w") as f:
        f.write("std::string noAnagramsWords[] = {\n")
        for word in words:
            f.write("\"" + word + "\",\n")
        f.write("};")
        
if __name__ == "__main__":
    main()