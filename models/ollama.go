package models

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/milvus"
)

const (
	milvusAddr     = "10.18.150.1:19530"
	ollamaAddr     = "http://192.168.1.228:11434"
	llmModel       = "qwen2.5:7b"
	embedderModel  = "nomic-embed-text:latest"
	dbName         = "default"
	collectionName = "text_collection"
	dim            = 128
	topK           = 5
	chunkSize      = 100
	chunkOverlap   = 0
	msgFmt         = "==== %s ====\n"
	scoreThreshold = 0.5 // 设置分数阈值
)

func OllamModel() (llms.Model, embeddings.Embedder, error) {
	llm, err := ollama.New(
		ollama.WithModel(llmModel),
		ollama.WithServerURL(ollamaAddr),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to initialize LLM: %w", err)
	}

	embedderModel, err := ollama.New(
		ollama.WithModel(embedderModel),
		ollama.WithServerURL(ollamaAddr),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to initialize embedder model: %w", err)
	}

	embedder, err := embeddings.NewEmbedder(embedderModel)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create embedder: %w", err)
	}

	return llm, embedder, nil
}
func MilvusStore(ctx context.Context, embedder embeddings.Embedder) (*milvus.Store, error) {
	idx, err := entity.NewIndexIvfFlat(entity.L2, dim)
	if err != nil {
		return nil, fmt.Errorf("failed to create ivf flat index: %w", err)
	}

	store, err := milvus.New(
		ctx,
		client.Config{
			Address: milvusAddr,
			DBName:  dbName,
		},
		milvus.WithEmbedder(embedder),
		milvus.WithCollectionName(collectionName),
		milvus.WithIndex(idx),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Milvus store: %w", err)
	}

	return &store, nil
}

func UseRetriever(store *milvus.Store, query string, topk int) ([]schema.Document, error) {
	retriever := vectorstores.ToRetriever(store, topk)
	//retriever := vectorstores.ToRetriever(store, topk, vectorstores.WithScoreThreshold(scoreThreshold))

	docRetrieved, err := retriever.GetRelevantDocuments(context.Background(), query)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve documents: %w", err)
	}
	return docRetrieved, nil
}

func GetAnswer(ctx context.Context, llm llms.Model, docs []schema.Document, query string) (string, error) {
	history := memory.NewChatMessageHistory()
	for _, doc := range docs {
		history.AddAIMessage(ctx, doc.PageContent)
	}

	conversation := memory.NewConversationBuffer(memory.WithChatHistory(history))
	executor := agents.NewExecutor(
		agents.NewConversationalAgent(llm, nil),
		agents.WithMemory(conversation),
	)

	options := []chains.ChainCallOption{chains.WithTemperature(0.8)}
	res, err := chains.Run(ctx, executor, query, options...)
	if err != nil {
		return "", fmt.Errorf("failed to generate answer: %w", err)
	}
	return res, nil
}
