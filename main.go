package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"ollama-milvus-vectorstore-example/config"
	"ollama-milvus-vectorstore-example/models"
	"ollama-milvus-vectorstore-example/utils"
)

func main() {
	// Initialize logger
	logger := log.New(os.Stdout, "[APP] ", log.LstdFlags|log.Lshortfile)

	// Load configuration
	start := time.Now()
	cfg, err := config.LoadConfig("config/config.yaml")
	if err != nil {
		logger.Fatalf("Failed to load config: %v", err)
	}
	logger.Printf("Config loaded in %v", time.Since(start))

	// Initialize model service
	modelService := models.NewModelService(cfg, logger)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Initialize models and vector store
	if err := modelService.Initialize(ctx); err != nil {
		logger.Fatalf("Initialization failed: %v", err)
	}

	// Add documents to vector store
	// if err := modelService.AddDocuments(ctx, "./index.txt"); err != nil {
	// 	logger.Fatalf("Document load failed: %v", err)
	// }

	// Main interaction loop
	for {
		query, err := utils.GetUserInput("请输入您的问题 (输入 'exit' 退出):")
		if err != nil {
			logger.Printf("Failed to get user input: %v", err)
			continue
		}

		if strings.ToLower(query) == "exit" {
			break
		}

		// Process query
		answer, err := modelService.Query(ctx, query, cfg.Processing.TopK)
		if err != nil {
			logger.Printf("Query processing failed: %v", err)
			continue
		}

		// Display results
		fmt.Println(strings.Repeat("#", 40))
		fmt.Println("回答:")
		fmt.Println(answer)
		fmt.Println(strings.Repeat("#", 40))
	}

	logger.Println("Application shutdown complete")
}
