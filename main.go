package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"ollama-milvus-vectorstore-example/api"
	"ollama-milvus-vectorstore-example/config"
	"ollama-milvus-vectorstore-example/models"
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

	// Create qa_pairs table if not exists
	// _, err = modelService.DB.ExecContext(ctx, `
	// 	CREATE TABLE IF NOT EXISTS qa_pairs (
	// 		id BIGINT AUTO_INCREMENT PRIMARY KEY,
	// 		question TEXT NOT NULL,
	// 		answer TEXT NOT NULL,
	// 		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	// 	)`)
	// if err != nil {
	// 	logger.Fatalf("Failed to create qa_pairs table: %v", err)
	// }

	// Test StoreQA
	// err = modelService.StoreQA(ctx, "开发环境日志系统地址?", "开发环境kibana访问地址是:https://kibana-dev.test.com/,账户密码使用ldap")
	// if err != nil {
	// 	logger.Printf("Failed to store QA pair: %v", err)
	// } else {
	// 	logger.Println("Successfully stored QA pair")
	// }

	// 创建路由器
	mux := http.NewServeMux()
	handler := api.NewHandler(modelService)

	// 添加中间件
	wrappedMux := api.CORSMiddleware(api.LoggingMiddleware(logger)(mux))

	// 设置路由
	mux.HandleFunc("/api/query", handler.QueryHandler)
	mux.HandleFunc("/api/store", handler.StoreQAHandler)
	mux.HandleFunc("/api/questions", handler.GetQuestionsHandler)
	mux.HandleFunc("/health", handler.HealthCheckHandler)

	// 静态文件服务
	fs := http.FileServer(http.Dir("static"))
	mux.Handle("/static/", http.StripPrefix("/static/", fs))

	// 启动 HTTP 服务器
	server := &http.Server{
		Addr:         ":8080",
		Handler:      wrappedMux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// 优雅关闭
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := server.Shutdown(ctx); err != nil {
			logger.Printf("HTTP server shutdown error: %v", err)
		}

		// 清理资源
		if err := modelService.Close(ctx); err != nil {
			logger.Printf("Cleanup error: %v", err)
		}
	}()

	logger.Printf("Starting HTTP server on %s", server.Addr)
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		logger.Fatalf("HTTP server error: %v", err)
	}

	// Main interaction loop
	// for {
	// 	query, err := utils.GetUserInput("请输入您的问题 (输入 'exit' 退出):")
	// 	if err != nil {
	// 		logger.Printf("Failed to get user input: %v", err)
	// 		continue
	// 	}

	// 	if strings.ToLower(query) == "exit" {
	// 		break
	// 	}

	// 	// Process query
	// 	answer, err := modelService.QueryWithRetrieve(ctx, query, cfg.Processing.TopK)
	// 	if err != nil {
	// 		logger.Printf("Query processing failed: %v", err)
	// 		continue
	// 	}

	// 	// Display results
	// 	fmt.Println(strings.Repeat("#", 40))
	// 	fmt.Println("回答:")
	// 	fmt.Println(answer)
	// 	fmt.Println(strings.Repeat("#", 40))
	// }

	logger.Println("Application shutdown complete")
}
