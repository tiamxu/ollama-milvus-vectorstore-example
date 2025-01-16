package vectorstore

import (
	"context"
	"fmt"
	"net/url"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

type QdrantConfig struct {
	Address    string `yaml:"address"`
	Collection string `yaml:"collection"`
}

func (c *QdrantConfig) Validate() error {
	if c.Address == "" {
		return fmt.Errorf("qdrant address is required")
	}
	return nil
}

type QdrantStore struct {
	store    *qdrant.Store
	embedder embeddings.Embedder
	cfg      *QdrantConfig
}

func NewQdrantStore(cfg *QdrantConfig, embedder embeddings.Embedder) *QdrantStore {
	return &QdrantStore{
		embedder: embedder,
		cfg:      cfg,
	}
}

func (q *QdrantStore) Initialize(ctx context.Context) error {
	qdrantURL, err := url.Parse(q.cfg.Address)
	if err != nil {
		return fmt.Errorf("invalid Qdrant URL: %w", err)
	}

	store, err := qdrant.New(
		qdrant.WithURL(*qdrantURL),
		qdrant.WithCollectionName(q.cfg.Collection),
		qdrant.WithEmbedder(q.embedder),
	)
	if err != nil {
		return fmt.Errorf("failed to initialize Qdrant store: %w", err)
	}

	q.store = &store
	return nil
}

func (q *QdrantStore) AddDocuments(ctx context.Context, docs []schema.Document) error {
	_, err := q.store.AddDocuments(ctx, docs)
	return err
}

// func (q *QdrantStore) SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error) {
// 	return q.store.SimilaritySearch(ctx, query, k)
// }

func (q *QdrantStore) Search(ctx context.Context, query string, k int) ([]schema.Document, error) {
	return q.store.SimilaritySearch(ctx, query, k)
}

func (q *QdrantStore) Close(ctx context.Context) error {
	return nil
}
