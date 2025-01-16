package models

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/schema"

	"ollama-milvus-vectorstore-example/config"
	"ollama-milvus-vectorstore-example/models/vectorstore"
	"ollama-milvus-vectorstore-example/utils"

	_ "github.com/go-sql-driver/mysql"
)

type ModelService struct {
	llm      llms.Model
	embedder embeddings.Embedder
	store    vectorstore.VectorStore
	cfg      *config.Config
	logger   *log.Logger
	db       *sql.DB
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

	// Initialize MySQL connection
	db, err := sql.Open("mysql", s.cfg.MySQL.DSN)
	if err != nil {
		return fmt.Errorf("failed to connect to MySQL: %w", err)
	}
	s.db = db

	return nil
}

// StoreQA stores question-answer pair
func (s *ModelService) StoreQA(ctx context.Context, question string, answer string) error {
	// 1. Store answer to MySQL
	result, err := s.db.ExecContext(ctx,
		"INSERT INTO qa_pairs (question, answer) VALUES (?, ?)",
		question, answer)
	if err != nil {
		return fmt.Errorf("failed to store answer: %w", err)
	}

	// 2. Get generated ID
	id, err := result.LastInsertId()
	if err != nil {
		return fmt.Errorf("failed to get last insert ID: %w", err)
	}

	// 3. Store question and ID to vector store
	doc := schema.Document{
		PageContent: question,
		Metadata: map[string]interface{}{
			"qa_id": id,
		},
	}

	err = s.store.AddDocuments(ctx, []schema.Document{doc})
	return err
}

// RetrieveAnswer retrieves answers for a query
func (s *ModelService) RetrieveAnswer(ctx context.Context, query string, topK int) ([]string, error) {
	// 1. Vector search for similar questions
	docs, err := s.retrieveDocuments(ctx, query, topK)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve documents: %w", err)
	}

	// 2. Extract IDs
	var ids []int64
	for _, doc := range docs {
		if id, ok := doc.Metadata["qa_id"]; ok {
			switch v := id.(type) {
			case int64:
				ids = append(ids, v)
			case float64:
				ids = append(ids, int64(v))
			default:
				return nil, fmt.Errorf("unexpected type for qa_id: %T", id)
			}
		}
	}

	// 3. Query answers from MySQL
	if len(ids) == 0 {
		return nil, nil
	}

	// Convert ids to comma-separated string
	idStrs := make([]string, len(ids))
	for i, id := range ids {
		idStrs[i] = fmt.Sprintf("%d", id)
	}
	idList := strings.Join(idStrs, ",")

	rows, err := s.db.QueryContext(ctx,
		fmt.Sprintf("SELECT answer FROM qa_pairs WHERE id IN (%s)", idList))
	if err != nil {
		return nil, fmt.Errorf("failed to query answers: %w", err)
	}
	defer rows.Close()

	// 4. Collect answers
	var answers []string
	for rows.Next() {
		var answer string
		if err := rows.Scan(&answer); err != nil {
			return nil, fmt.Errorf("failed to scan answer: %w", err)
		}
		answers = append(answers, answer)
	}

	return answers, nil
}

func (s *ModelService) initModels() (llms.Model, embeddings.Embedder, error) {
	// Create HTTP client with timeout and retry
	httpClient := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        10,
			IdleConnTimeout:     30 * time.Second,
			DisableCompression:  true,
			MaxIdleConnsPerHost: 10,
		},
	}

	llm, err := ollama.New(
		ollama.WithModel(s.cfg.Ollama.LLMModel),
		ollama.WithServerURL(s.cfg.Ollama.Address),
		ollama.WithHTTPClient(httpClient),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to initialize LLM: %w", err)
	}

	embedderModel, err := ollama.New(
		ollama.WithModel(s.cfg.Ollama.EmbedderModel),
		ollama.WithServerURL(s.cfg.Ollama.Address),
		ollama.WithHTTPClient(httpClient),
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

func (s *ModelService) initVectorStore(ctx context.Context, embedder embeddings.Embedder) (vectorstore.VectorStore, error) {
	switch s.cfg.VectorStore.Type {
	case "milvus":
		milvusCfg := &vectorstore.MilvusConfig{
			Address:    s.cfg.VectorStore.Milvus.Address,
			DBName:     s.cfg.VectorStore.Milvus.DBName,
			Collection: s.cfg.VectorStore.Milvus.Collection,
			IndexType:  s.cfg.VectorStore.Milvus.Index.Type,
		}
		store := vectorstore.NewMilvusStore(milvusCfg, embedder)
		return store, store.Initialize(ctx)

	case "qdrant":
		qdrantCfg := &vectorstore.QdrantConfig{
			Address:    s.cfg.VectorStore.Qdrant.Address,
			Collection: s.cfg.VectorStore.Qdrant.Collection,
		}
		store := vectorstore.NewQdrantStore(qdrantCfg, embedder)
		return store, store.Initialize(ctx)

	default:
		return nil, fmt.Errorf("unsupported vector store type: %s", s.cfg.VectorStore.Type)
	}
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

// generateFinalResponse generates a final response using LLM based on question and answers
func (s *ModelService) generateFinalResponse(ctx context.Context, query string, answers []string) (string, error) {
	// Create prompt with question and answers
	prompt := fmt.Sprintf("根据以下问题和相关答案，生成一个完整的回答：\n\n问题：%s\n\n相关答案：\n%s",
		query, strings.Join(answers, "\n\n"))

	// Generate response using LLM
	res, err := s.llm.GenerateContent(ctx, []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart(prompt)},
		},
	})
	if err != nil {
		return "", fmt.Errorf("failed to generate final response: %w", err)
	}

	return res.Choices[0].Content, nil
}

// QueryWithRetrieve uses RetrieveAnswer to get answers from database and generate final response
func (s *ModelService) QueryWithRetrieve(ctx context.Context, query string, topK int) (string, error) {
	start := time.Now()
	defer func() {
		s.logger.Printf("QueryWithRetrieve completed in %v", time.Since(start))
	}()

	// Retrieve answers from database
	answers, err := s.RetrieveAnswer(ctx, query, topK)
	if err != nil {
		return "", fmt.Errorf("failed to retrieve answers: %w", err)
	}

	// If no answers found, return default response
	if len(answers) == 0 {
		return "没有找到相关答案", nil
	}

	// Generate final response using LLM
	return s.generateFinalResponse(ctx, query, answers)
}

func (s *ModelService) AddDocuments(ctx context.Context, fileName string) error {
	// Load and split documents
	docs := utils.TextToChunks(fileName, s.cfg.Processing.ChunkSize, s.cfg.Processing.ChunkOverlap)

	// Add documents to vector store
	err := s.store.AddDocuments(ctx, docs)
	if err != nil {
		return fmt.Errorf("failed to add documents: %w", err)
	}

	s.logger.Println("Documents successfully indexed")
	return nil
}

func (s *ModelService) retrieveDocuments(ctx context.Context, query string, topK int) ([]schema.Document, error) {
	// 使用 Search 方法替代 SimilaritySearch
	docs, err := s.store.Search(ctx, query, topK)
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

func (s *ModelService) Close(ctx context.Context) error {
	var errs []error

	if s.store != nil {
		if err := s.store.Close(ctx); err != nil {
			errs = append(errs, fmt.Errorf("failed to close vector store: %w", err))
		}
	}

	if s.db != nil {
		if err := s.db.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close database: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors during cleanup: %v", errs)
	}
	return nil
}
