package mcp

import (
	"encoding/json"
	"fmt"
	"net/http"
)

const (
	ProviderOllama       = "ollama"
	DefaultOllamaBaseURL = "https://ollama.com"
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

// buildUrl returns the Ollama native API endpoint
func (oc *OllamaClient) buildUrl() string {
	return oc.BaseURL + "/api/chat"
}

// buildMCPRequestBody builds the request body for Ollama native API format
func (oc *OllamaClient) buildMCPRequestBody(systemPrompt, userPrompt string) map[string]any {
	// Build messages array
	messages := []map[string]string{}

	// If system prompt exists, add system message
	if systemPrompt != "" {
		messages = append(messages, map[string]string{
			"role":    "system",
			"content": systemPrompt,
		})
	}
	// Add user message
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

// parseMCPResponse parses Ollama native API response
func (oc *OllamaClient) parseMCPResponse(body []byte) (string, error) {
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