package vectorstore

import (
	"context"
	"fmt"
	"net/url"

	qdrantclient "github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

type QdrantConfig struct {
	Address    string `yaml:"address"`
	Host       string `yaml:"host"`
	Port       int    `yaml:"port"`
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
	client, err := qdrantclient.NewClient(&qdrantclient.Config{
		Host:                   "120.24.61.231",
		Port:                   16334,
		APIKey:                 "ZfYOjrdr2io25WUKvpdwnJ8gfvc",
		UseTLS:                 false,
		SkipCompatibilityCheck: true,
	})
	if err != nil {
		return fmt.Errorf("failed to create qdrant client: %w", err)
	}
	exist, err := client.CollectionExists(ctx, q.cfg.Collection)
	if err != nil {
		return fmt.Errorf("failed to check collection exists: %w", err)
	}
	if !exist {
		err = client.CreateCollection(ctx, &qdrantclient.CreateCollection{
			CollectionName: q.cfg.Collection,
			VectorsConfig: qdrantclient.NewVectorsConfig(&qdrantclient.VectorParams{
				Size:     1024,
				Distance: qdrantclient.Distance_Cosine,
			}),
		})
		if err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
	}

	store, err := qdrant.New(
		qdrant.WithURL(*qdrantURL),
		qdrant.WithCollectionName(q.cfg.Collection),
		qdrant.WithEmbedder(q.embedder),
		qdrant.WithAPIKey("ZfYOjrdr2io25WUKvpdwnJ8gfvc"),
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
