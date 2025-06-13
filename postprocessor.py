def isPangram(indices):
    total = 0
    for i in indices:
        total = total | i
    return total == 67108863

def main():
    with open("include/filteredWords.inl", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    word_indices = [26, 7647, 7668, 7949, 8000, 8001]

    for i in word_indices:
        print(lines[i + 1])

    print("Is pangram? ", isPangram(word_indices))
        
if __name__ == "__main__":
    main()