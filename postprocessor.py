import pprint as pp

def findPangrams(start):
    with open("allsolutions.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [(line if line.split(",")[0] == start else "a") for line in lines]

    lines.sort()
    with open(f"allsolutions_{start}.txt", "w") as f:
        for line in lines:
            if line != "a":
                f.write(line + "\n")

def findNoPangramWords():
    with open("allsolutions.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(",") for line in lines]
        
    print("All solutions read")
    with open("words.txt", "r") as f:
        words = f.readlines()
        words = set(words)
    print("All words read")

    wordSet = set()
    for line in lines:
        wordSet.add(line[0])
        wordSet.add(line[1])
        wordSet.add(line[2])
        wordSet.add(line[3])
        wordSet.add(line[4])
        wordSet.add(line[5])
    print("All solutions added to set")

    noPangramWords = words - wordSet
    noPangramWords = list(noPangramWords)
    noPangramWords.sort()
    with open("noPangramWords.txt", "w") as f:
        for word in noPangramWords:
            f.write(word)
    
def main():
    # findPangrams("blonx")
    findNoPangramWords()
        
if __name__ == "__main__":
    main()