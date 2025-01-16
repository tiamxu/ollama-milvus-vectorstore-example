package api

import (
	"net/http"
	"sync"
	"time"
)

type RateLimiter struct {
	requests map[string][]time.Time
	mu       sync.Mutex
	limit    int
	window   time.Duration
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		requests: make(map[string][]time.Time),
		limit:    limit,
		window:   window,
	}
}

func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := r.RemoteAddr

		rl.mu.Lock()
		now := time.Now()

		// 清理过期的请求记录
		windowStart := now.Add(-rl.window)
		var recent []time.Time
		for _, t := range rl.requests[ip] {
			if t.After(windowStart) {
				recent = append(recent, t)
			}
		}

		if len(recent) >= rl.limit {
			rl.mu.Unlock()
			jsonResponse(w, http.StatusTooManyRequests, Response{
				Success: false,
				Error:   "请求过于频繁，请稍后再试",
			})
			return
		}

		rl.requests[ip] = append(recent, now)
		rl.mu.Unlock()

		next.ServeHTTP(w, r)
	})
}
