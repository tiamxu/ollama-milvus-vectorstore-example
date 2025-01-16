package vectorstore

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/milvus"
)

type MilvusConfig struct {
	Address    string
	DBName     string
	Collection string
	IndexType  string
}

func (c *MilvusConfig) Validate() error {
	if c.Address == "" {
		return fmt.Errorf("milvus address is required")
	}
	if c.Collection == "" {
		return fmt.Errorf("milvus collection name is required")
	}
	if c.IndexType == "" {
		return fmt.Errorf("milvus index type is required")
	}
	return nil
}

type MilvusStore struct {
	store    *milvus.Store
	embedder embeddings.Embedder
	cfg      *MilvusConfig
}

func NewMilvusStore(cfg *MilvusConfig, embedder embeddings.Embedder) *MilvusStore {
	return &MilvusStore{
		embedder: embedder,
		cfg:      cfg,
	}
}

func (m *MilvusStore) Initialize(ctx context.Context) error {
	var idx entity.Index
	var err error

	switch m.cfg.IndexType {
	case "IVF_FLAT":
		idx, err = entity.NewIndexIvfFlat(entity.L2, 768)
	case "IVF_SQ8":
		idx, err = entity.NewIndexIvfSQ8(entity.L2, 768)
	case "HNSW":
		idx, err = entity.NewIndexHNSW(entity.L2, 16, 200)
	default:
		return fmt.Errorf("unsupported index type: %s", m.cfg.IndexType)
	}

	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	store, err := milvus.New(
		ctx,
		client.Config{
			Address: m.cfg.Address,
			DBName:  m.cfg.DBName,
		},
		milvus.WithEmbedder(m.embedder),
		milvus.WithCollectionName(m.cfg.Collection),
		milvus.WithIndex(idx),
	)
	if err != nil {
		return fmt.Errorf("failed to initialize Milvus store: %w", err)
	}

	m.store = &store
	return nil
}

func (m *MilvusStore) AddDocuments(ctx context.Context, docs []schema.Document) error {
	_, err := m.store.AddDocuments(ctx, docs)
	return err
}

// func (m *MilvusStore) SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error) {
// 	return m.store.SimilaritySearch(ctx, query, k)
// }

func (m *MilvusStore) Search(ctx context.Context, query string, k int) ([]schema.Document, error) {
	return m.store.SimilaritySearch(ctx, query, k)
}

func (m *MilvusStore) Close(ctx context.Context) error {
	return nil
}
