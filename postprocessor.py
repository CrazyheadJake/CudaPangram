import pprint as pp

def expandCachedWords():
    with open("filteredWords.txt", "r") as f:
        no_anagram_words = f.readlines()
        no_anagram_words = [line.strip() for line in no_anagram_words]

    with open("solutions.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.strip(",") for line in lines]
        lines = [[int(word) for word in line.split(", ")] for line in lines]

    cache = {}
    fullSolutions = []
    for i, line in enumerate(lines):
        for k, word in enumerate(line):
            if (word > len(no_anagram_words)):
                break
            else:
                lines[i][k] = no_anagram_words[word]
            if (k == 5):
                fullSolutions.append(line)
    
    for i, soln in enumerate(fullSolutions):
        for k, word in enumerate(soln):
            pass
    


    pp.pprint(lines)
    print("Total Full Solutions: ", len(fullSolutions))

def main():

    expandCachedWords()
        
if __name__ == "__main__":
    main()