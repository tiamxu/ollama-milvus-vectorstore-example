package models

import (
	"context"
	"fmt"
	"log"
	"time"

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

	"ollama-milvus-vectorstore-example/config"
	"ollama-milvus-vectorstore-example/utils"
)

type ModelService struct {
	llm      llms.Model
	embedder embeddings.Embedder
	store    *milvus.Store
	cfg      *config.Config
	logger   *log.Logger
}

func NewModelService(cfg *config.Config, logger *log.Logger) *ModelService {
	return &ModelService{
		cfg:    cfg,
		logger: logger,
	}
}

func (s *ModelService) Initialize(ctx context.Context) error {
	start := time.Now()
	defer func() {
		s.logger.Printf("Model initialization completed in %v", time.Since(start))
	}()

	// Initialize LLM and Embedder
	llm, embedder, err := s.initModels()
	if err != nil {
		return fmt.Errorf("model initialization failed: %w", err)
	}

	// Initialize Vector Store
	store, err := s.initVectorStore(ctx, embedder)
	if err != nil {
		return fmt.Errorf("vector store initialization failed: %w", err)
	}

	s.llm = llm
	s.embedder = embedder
	s.store = store

	return nil
}

func (s *ModelService) initModels() (llms.Model, embeddings.Embedder, error) {
	llm, err := ollama.New(
		ollama.WithModel(s.cfg.Ollama.LLMModel),
		ollama.WithServerURL(s.cfg.Ollama.Address),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to initialize LLM: %w", err)
	}

	embedderModel, err := ollama.New(
		ollama.WithModel(s.cfg.Ollama.EmbedderModel),
		ollama.WithServerURL(s.cfg.Ollama.Address),
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

func (s *ModelService) initVectorStore(ctx context.Context, embedder embeddings.Embedder) (*milvus.Store, error) {
	idx, err := entity.NewIndexIvfFlat(entity.L2, 128)
	if err != nil {
		return nil, fmt.Errorf("failed to create ivf flat index: %w", err)
	}

	store, err := milvus.New(
		ctx,
		client.Config{
			Address: s.cfg.Milvus.Address,
			DBName:  s.cfg.Milvus.DBName,
		},
		milvus.WithEmbedder(embedder),
		milvus.WithCollectionName(s.cfg.Milvus.Collection),
		milvus.WithIndex(idx),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Milvus store: %w", err)
	}
	// docs := utils.TextToChunks("./index.txt", 50, 0)
	// _, err = store.AddDocuments(ctx, docs)
	// if err != nil {
	// 	log.Fatalf("AddDocument: %v\n", err)
	// }
	return &store, nil
}

func (s *ModelService) Query(ctx context.Context, query string, topK int) (string, error) {
	start := time.Now()
	defer func() {
		s.logger.Printf("Query completed in %v", time.Since(start))
	}()

	// Retrieve relevant documents
	docs, err := s.retrieveDocuments(ctx, query, topK)
	if err != nil {
		return "", fmt.Errorf("document retrieval failed: %w", err)
	}

	// Generate answer
	answer, err := s.generateAnswer(ctx, docs, query)
	if err != nil {
		return "", fmt.Errorf("answer generation failed: %w", err)
	}

	return answer, nil
}

func (s *ModelService) AddDocuments(ctx context.Context, fileName string) error {
	// Load and split documents
	docs := utils.TextToChunks(fileName, s.cfg.Processing.ChunkSize, s.cfg.Processing.ChunkOverlap)

	// Add documents to vector store
	_, err := s.store.AddDocuments(ctx, docs)
	if err != nil {
		return fmt.Errorf("failed to add documents: %w", err)
	}

	s.logger.Println("Documents successfully indexed")
	return nil
}

func (s *ModelService) retrieveDocuments(ctx context.Context, query string, topK int) ([]schema.Document, error) {
	retriever := vectorstores.ToRetriever(s.store, topK)
	docs, err := retriever.GetRelevantDocuments(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve documents: %w", err)
	}
	return docs, nil
}

func (s *ModelService) generateAnswer(ctx context.Context, docs []schema.Document, query string) (string, error) {
	history := memory.NewChatMessageHistory()
	for _, doc := range docs {
		history.AddAIMessage(ctx, doc.PageContent)
	}

	conversation := memory.NewConversationBuffer(memory.WithChatHistory(history))
	executor := agents.NewExecutor(
		agents.NewConversationalAgent(s.llm, nil),
		agents.WithMemory(conversation),
	)

	res, err := chains.Run(ctx, executor, query)
	if err != nil {
		return "", fmt.Errorf("failed to generate answer: %w", err)
	}
	return res, nil
}
