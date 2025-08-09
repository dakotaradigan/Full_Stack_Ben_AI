# Function Routing Workflow Updates

## Overview
Updated the AI assistant's function routing logic to optimize benchmark discovery and exploration for financial advisors.

## Key Changes Made

### 1. New Function Routing Workflow
Implemented a clear 4-step decision logic:

- **Minimum Check**: "What's the minimum for [benchmark]?" → Use `get_minimum`
- **Blend Calculation**: "What's the minimum for X% A + Y% B?" → Use `blend_minimum`
- **Alternatives Request**: "Alternatives to [benchmark]" or "Similar to [benchmark]" → Use `search_by_characteristics`
- **General Search**: "Find benchmarks with [criteria]" → Use `search_benchmarks`

### 2. Enhanced Vector Database Integration
- `search_by_characteristics` now leverages both:
  - **Vector similarity search**: Uses semantic embeddings to find similar benchmarks
  - **Structured metadata filtering**: Applies precise filters for region, asset class, style, factors, ESG
  - **Fast and accurate**: Gets 5-10 most relevant results quickly from vector database

### 3. Removed Portfolio Size Requirements
- Eliminated the requirement to ask users for portfolio size upfront
- Focus shifted to exploration and discovery
- Advisors can naturally explore options and hone in on the right benchmark
- Portfolio size filtering only used when explicitly provided by user

### 4. Tool Selection Optimization
Updated tool selection guidelines:
- `get_minimum`: Direct database lookup for specific benchmark minimums
- `blend_minimum`: Portfolio blend calculations
- `search_by_characteristics`: **Primary tool** - combines vector search + structured filtering for alternatives
- `search_benchmarks`: Pure vector search for general exploration and criteria-based searches
- `search_viable_alternatives`: Only when user explicitly provides portfolio size constraints

## Technical Implementation

### Files Modified
1. **system_prompt.txt**: Updated workflow instructions and removed portfolio size requirements
2. **chatbot.py**: No changes needed - existing functions already support new workflow

### Function Capabilities Confirmed
- `search_by_characteristics` properly calls vector database through `search_benchmarks`
- Combines semantic search with structured metadata filtering
- Maintains fast performance while ensuring accuracy

## Benefits
1. **Improved User Experience**: No forced portfolio size collection
2. **Better Discovery**: Leverages vector database for semantic similarity
3. **Precise Filtering**: Structured metadata ensures characteristic matching
4. **Advisor-Friendly**: Supports natural exploration workflow
5. **Optimal Performance**: Uses the right tool for each query type

## Usage Examples
- "Alternatives to Russell 2000" → `search_by_characteristics` finds small-cap alternatives
- "ESG options in US markets" → `search_benchmarks` with criteria-based search  
- "What's the minimum for S&P 500?" → `get_minimum` for direct lookup
- "60% S&P 500, 40% EAFE minimum" → `blend_minimum` for calculations