package main

import (
	"context"
	"fmt"
	"log"
	"ollama-milvus-vectorstore-example/models"
	"ollama-milvus-vectorstore-example/utils"
	"strings"
)

const (
	topK           = 3
	chunkSize      = 50
	chunkOverlap   = 0
	msgFmt         = "==== %s ====\n"
	scoreThreshold = 0.5 // 设置分数阈值
)

func main() {

	ctx := context.Background()

	// Initialize models and Milvus store
	llm, embedder, err := models.OllamModel()
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if closer, ok := llm.(interface{ Close() error }); ok {
			if err := closer.Close(); err != nil {
				log.Printf("Failed to close LLM: %v", err)
			}
		}
	}()

	store, err := models.MilvusStore(ctx, embedder)
	if err != nil {
		log.Fatal(err)
	}
	// docs := utils.TextToChunks("./index.txt", chunkSize, chunkOverlap)
	// _, err = store.AddDocuments(ctx, docs)
	// if err != nil {
	// 	log.Fatalf("AddDocument: %v\n", err)
	// }
	// Get user input and perform retrieval
	query, err := utils.GetUserInput("输入问题")
	if err != nil {
		log.Fatal(err)
	}

	docRetrieved, err := models.UseRetriever(store, query, topK)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Retrieved Documents:\n%v\n", docRetrieved)

	// Generate answer based on retrieved documents
	answer, err := models.GetAnswer(ctx, llm, docRetrieved, query)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(strings.Repeat("#", 40))
	fmt.Println(answer)
}
