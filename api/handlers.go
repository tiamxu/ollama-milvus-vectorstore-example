package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"ollama-milvus-vectorstore-example/models"
)

type Handler struct {
	modelService *models.ModelService
}

func NewHandler(modelService *models.ModelService) *Handler {
	return &Handler{
		modelService: modelService,
	}
}

type QueryRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k,omitempty"`
}

type QAPairRequest struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

type Response struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

func (h *Handler) QueryHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		jsonResponse(w, http.StatusMethodNotAllowed, Response{Success: false, Error: "仅支持 POST 方法"})
		return
	}

	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonResponse(w, http.StatusBadRequest, Response{Success: false, Error: "无效的请求体"})
		return
	}

	if req.Query == "" {
		jsonResponse(w, http.StatusBadRequest, Response{Success: false, Error: "查询内容不能为空"})
		return
	}

	if req.TopK <= 0 {
		req.TopK = 5 // 默认值
	} else if req.TopK > 20 {
		req.TopK = 20 // 限制最大值
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	answer, err := h.modelService.QueryWithRetrieve(ctx, req.Query, req.TopK)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			jsonResponse(w, http.StatusGatewayTimeout, Response{Success: false, Error: "请求超时"})
			return
		}
		jsonResponse(w, http.StatusInternalServerError, Response{Success: false, Error: err.Error()})
		return
	}

	jsonResponse(w, http.StatusOK, Response{Success: true, Data: answer})
}

func (h *Handler) StoreQAHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		jsonResponse(w, http.StatusMethodNotAllowed, Response{Success: false, Error: "仅支持 POST 方法"})
		return
	}

	var req QAPairRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonResponse(w, http.StatusBadRequest, Response{Success: false, Error: "无效的请求体"})
		return
	}

	err := h.modelService.StoreQA(r.Context(), req.Question, req.Answer)
	if err != nil {
		jsonResponse(w, http.StatusInternalServerError, Response{Success: false, Error: err.Error()})
		return
	}

	jsonResponse(w, http.StatusOK, Response{Success: true})
}

func (h *Handler) GetQuestionsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		jsonResponse(w, http.StatusMethodNotAllowed, Response{Success: false, Error: "仅支持 GET 方法"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	questions, err := h.modelService.GetStoredQuestions(ctx)
	if err != nil {
		jsonResponse(w, http.StatusInternalServerError, Response{Success: false, Error: err.Error()})
		return
	}

	jsonResponse(w, http.StatusOK, Response{Success: true, Data: questions})
}

func jsonResponse(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
