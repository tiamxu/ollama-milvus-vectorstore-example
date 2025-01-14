package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Milvus     MilvusConfig     `yaml:"milvus"`
	Ollama     OllamaConfig     `yaml:"ollama"`
	Processing ProcessingConfig `yaml:"processing"`
}

type MilvusConfig struct {
	Address    string      `yaml:"address"`
	DBName     string      `yaml:"db_name"`
	Collection string      `yaml:"collection"`
	Index      IndexConfig `yaml:"index"`
}

type OllamaConfig struct {
	Address       string  `yaml:"address"`
	LLMModel      string  `yaml:"llm_model"`
	EmbedderModel string  `yaml:"embedder_model"`
	Temperature   float64 `yaml:"temperature"`
}

type ProcessingConfig struct {
	ChunkSize      int     `yaml:"chunk_size"`
	ChunkOverlap   int     `yaml:"chunk_overlap"`
	TopK           int     `yaml:"top_k"`
	ScoreThreshold float64 `yaml:"score_threshold"`
}

type IndexConfig struct {
	Type       string `yaml:"type"`
	MetricType string `yaml:"metric_type"`
	NList      int    `yaml:"nlist"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	if err := validateConfig(&cfg); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return &cfg, nil
}

func validateConfig(cfg *Config) error {
	if cfg.Milvus.Address == "" {
		return fmt.Errorf("milvus address is required")
	}
	if cfg.Ollama.Address == "" {
		return fmt.Errorf("ollama address is required")
	}
	if cfg.Processing.ChunkSize <= 0 {
		return fmt.Errorf("chunk size must be positive")
	}
	return nil
}
