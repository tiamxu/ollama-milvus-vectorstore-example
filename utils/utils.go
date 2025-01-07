package utils

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
)

func GetUserInput(prompt string) (string, error) {
	fmt.Printf("==== %s ====\n", prompt)
	reader := bufio.NewReader(os.Stdin)
	input, err := reader.ReadString('\n')
	if err != nil {
		return "", fmt.Errorf("failed to read input: %w", err)
	}
	return strings.TrimSpace(input), nil
}
func TextToChunks(dirFile string, chunkSize, chunkOverlap int) []schema.Document {
	file, err := os.Open(dirFile)
	if err != nil {
		log.Fatalf("failed opening file: %s", err)
		return nil
	}

	docLoaded := documentloaders.NewText(file)

	split := textsplitter.NewRecursiveCharacter()
	split.ChunkSize = chunkSize
	split.ChunkOverlap = chunkOverlap
	docs, err := docLoaded.LoadAndSplit(context.Background(), split)
	if err != nil {
		log.Fatalf("failed splitting text: %s", err)
	}

	log.Println("Document loaded:", len(docs))

	return docs
}
