package mcp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

const (
	ProviderOllama       = "ollama"
	DefaultOllamaBaseURL = "https://api.ollama.com/v1"
	DefaultOllamaModel   = "glm-4.7:cloud"
)

type OllamaClient struct {
	*Client
}

// NewOllamaClientWithOptions creates Ollama cloud client
//
// Usage examples:
//   // Basic usage
//   client := mcp.NewOllamaClientWithOptions()
//
//   // Custom configuration
//   client := mcp.NewOllamaClientWithOptions(
//       mcp.WithAPIKey("sk-xxx"),
//       mcp.WithLogger(customLogger),
//       mcp.WithTimeout(60*time.Second),
//   )
func NewOllamaClientWithOptions(opts ...ClientOption) AIClient {
	// 1. Create Ollama preset options
	ollamaOpts := []ClientOption{
		WithProvider(ProviderOllama),
		WithModel(DefaultOllamaModel),
		WithBaseURL(DefaultOllamaBaseURL),
	}

	// 2. Merge user options (user options have higher priority)
	allOpts := append(ollamaOpts, opts...)

	// 3. Create base client
	baseClient := NewClient(allOpts...).(*Client)

	// 4. Create Ollama client
	ollamaClient := &OllamaClient{
		Client: baseClient,
	}

	// 5. Set hooks to point to OllamaClient (implement dynamic dispatch)
	baseClient.hooks = ollamaClient

	return ollamaClient
}

func (oc *OllamaClient) SetAPIKey(apiKey string, customURL string, customModel string) {
	oc.APIKey = apiKey

	if len(apiKey) > 8 {
		oc.logger.Infof("🔧 [MCP] Ollama API Key: %s...%s", apiKey[:4], apiKey[len(apiKey)-4:])
	}
	if customURL != "" {
		oc.BaseURL = customURL
		oc.logger.Infof("🔧 [MCP] Ollama using custom BaseURL: %s", customURL)
	} else {
		oc.logger.Infof("🔧 [MCP] Ollama using default BaseURL: %s", oc.BaseURL)
	}
	if customModel != "" {
		oc.Model = customModel
		oc.logger.Infof("🔧 [MCP] Ollama using custom Model: %s", customModel)
	} else {
		oc.logger.Infof("🔧 [MCP] Ollama using default Model: %s", oc.Model)
	}
}

func (oc *OllamaClient) setAuthHeader(reqHeaders http.Header) {
	// Ollama uses Bearer token authentication
	reqHeaders.Set("Authorization", "Bearer "+oc.APIKey)
}

// buildUrl builds the appropriate API endpoint based on the base URL
// - For https://ollama.com: use /api/chat (native format)
// - For https://api.ollama.com: use /v1/chat/completions (OpenAI-compatible)
// - For custom URLs: detect format based on URL pattern
func (oc *OllamaClient) buildUrl() string {
	baseURL := oc.BaseURL

	// Check if using Ollama native API format
	if baseURL == "https://ollama.com" || baseURL == "http://ollama.com" {
		return baseURL + "/api/chat"
	}

	// Check if URL ends with /ollama.com (native format)
	if strings.HasSuffix(baseURL, "ollama.com") || strings.HasSuffix(baseURL, "ollama.com/") {
		// Remove trailing slash if present
		baseURL = strings.TrimSuffix(baseURL, "/")
		return baseURL + "/api/chat"
	}

	// Default: use OpenAI-compatible format (baseURL + /chat/completions)
	return baseURL + "/chat/completions"
}

// buildMCPRequestBody builds the request body for Ollama API
// Supports both native and OpenAI-compatible formats
func (oc *OllamaClient) buildMCPRequestBody(systemPrompt, userPrompt string) map[string]any {
	baseURL := oc.BaseURL

	// Check if using Ollama native API format
	isNativeFormat := baseURL == "https://ollama.com" ||
		baseURL == "http://ollama.com" ||
		strings.HasSuffix(baseURL, "ollama.com") ||
		strings.HasSuffix(baseURL, "ollama.com/")

	if isNativeFormat {
		// Ollama native format
		messages := []map[string]string{}
		if systemPrompt != "" {
			messages = append(messages, map[string]string{
				"role":    "system",
				"content": systemPrompt,
			})
		}
		messages = append(messages, map[string]string{
			"role":    "user",
			"content": userPrompt,
		})

		return map[string]any{
			"model":    oc.Model,
			"messages": messages,
			"stream":   false,
		}
	}

	// OpenAI-compatible format (default)
	// Use base client's implementation
	return oc.Client.buildMCPRequestBody(systemPrompt, userPrompt)
}

// parseMCPResponse parses the response from Ollama API
// Supports both native and OpenAI-compatible formats
func (oc *OllamaClient) parseMCPResponse(body []byte) (string, error) {
	baseURL := oc.BaseURL

	// Check if using Ollama native API format
	isNativeFormat := baseURL == "https://ollama.com" ||
		baseURL == "http://ollama.com" ||
		strings.HasSuffix(baseURL, "ollama.com") ||
		strings.HasSuffix(baseURL, "ollama.com/")

	if isNativeFormat {
		// Ollama native format response: {"message": {"content": "..."}, "done": true}
		var result struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Done bool `json:"done"`
		}

		if err := json.Unmarshal(body, &result); err != nil {
			return "", fmt.Errorf("failed to parse Ollama response: %w", err)
		}

		if result.Message.Content == "" {
			return "", fmt.Errorf("Ollama returned empty response")
		}

		return result.Message.Content, nil
	}

	// OpenAI-compatible format (default)
	// Use base client's implementation
	return oc.Client.parseMCPResponse(body)
}