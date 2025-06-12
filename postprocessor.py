def main():
    with open("include/filteredWords.inl", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    word_indices = [5, 4376, 4761, 5628, 6350, 8000]

    for i in word_indices:
        print(lines[i + 1])
        
if __name__ == "__main__":
    main()