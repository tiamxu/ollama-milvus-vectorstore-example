package api

import (
	"net/http"
	"runtime"
)

type HealthStatus struct {
	Status     string `json:"status"`
	Version    string `json:"version"`
	Memory     uint64 `json:"memory"`
	Goroutines int    `json:"goroutines"`
}

func (h *Handler) HealthCheckHandler(w http.ResponseWriter, r *http.Request) {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	status := HealthStatus{
		Status:     "healthy",
		Version:    "1.0.0", // 从配置或构建信息中获取
		Memory:     mem.Alloc,
		Goroutines: runtime.NumGoroutine(),
	}

	jsonResponse(w, http.StatusOK, Response{Success: true, Data: status})
}
