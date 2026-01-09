# HMLS Architecture Diagrams

## System Architecture Overview

```mermaid
graph TB
    subgraph "User API Layer"
        API[MagneticFieldInterpolator API]
    end
    
    subgraph "Factory Layer"
        Factory[DefaultInterpolatorFactory]
    end
    
    subgraph "Adapter Layer"
        CPUAdapter[CPUHermiteMLSInterpolatorAdapter]
        GPUAdapter[GPUHermiteMLSInterpolatorAdapter]
    end
    
    subgraph "Core Implementation"
        CPUCore[HermiteMLSInterpolator CPU]
        GPUKernel[CUDA HMLS Kernel]
    end
    
    subgraph "Supporting Components"
        KDTree[KD-Tree for k-NN Search]
        SpatialGrid[GPU Spatial Grid]
        MemMgr[GPU Memory Manager]
    end
    
    API --> Factory
    Factory --> CPUAdapter
    Factory --> GPUAdapter
    
    CPUAdapter --> CPUCore
    GPUAdapter --> CPUCore
    GPUAdapter --> GPUKernel
    
    CPUCore --> KDTree
    GPUKernel --> SpatialGrid
    GPUAdapter --> MemMgr
```

## HMLS Algorithm Flow

```mermaid
flowchart TD
    Start[Query Point q] --> FindNeighbors[Find k-Nearest Neighbors using KD-Tree]
    
    FindNeighbors --> ComputeWeights[Compute Weights w_i for each neighbor]
    
    ComputeWeights --> BuildBasis[Build Polynomial Basis Functions phi_j]
    
    BuildBasis --> AssembleSystem[Assemble Weighted Least Squares System]
    
    AssembleSystem --> AddValueConstraints[Add Value Constraints: p_i = f_i]
    
    AddValueConstraints --> AddDerivConstraints[Add Derivative Constraints: nabla_p_i = nabla_f_i]
    
    AddDerivConstraints --> SolveSystem[Solve Linear System for Coefficients]
    
    SolveSystem --> EvalPoly[Evaluate Polynomial at Query Point]
    
    EvalPoly --> Result[Return Interpolated Field & Derivatives]
```

## Data Flow - CPU Implementation

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Adapter
    participant HMLS
    participant KDTree
    
    User->>API: Query at point q
    API->>Adapter: query q
    Adapter->>HMLS: query q
    HMLS->>KDTree: findKNearest q, k
    KDTree-->>HMLS: neighbor indices & distances
    HMLS->>HMLS: computeWeights
    HMLS->>HMLS: buildLeastSquaresSystem
    HMLS->>HMLS: solveLeastSquares
    HMLS->>HMLS: evaluatePolynomial
    HMLS-->>Adapter: InterpolationResult
    Adapter-->>API: InterpolationResult
    API-->>User: Field values & derivatives
```

## Data Flow - GPU Implementation

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Adapter
    participant MemMgr
    participant Kernel
    participant SpatialGrid
    
    User->>API: QueryBatch query_points
    API->>Adapter: queryBatch query_points
    Adapter->>MemMgr: Allocate device memory
    Adapter->>MemMgr: Copy query_points to GPU
    Adapter->>Kernel: Launch HMLS kernel
    Kernel->>SpatialGrid: Find neighbors for each query
    SpatialGrid-->>Kernel: Neighbor data
    Kernel->>Kernel: Compute weights parallel
    Kernel->>Kernel: Build systems parallel
    Kernel->>Kernel: Solve systems parallel
    Kernel->>Kernel: Evaluate polynomials parallel
    Kernel-->>Adapter: Results in device memory
    Adapter->>MemMgr: Copy results to host
    MemMgr-->>Adapter: Host results
    Adapter-->>API: vector of InterpolationResult
    API-->>User: Batch results
```

## Class Hierarchy

```mermaid
classDiagram
    class IInterpolator {
        <<interface>>
        +query(Point3D) InterpolationResult
        +queryBatch(vector) vector
        +supportsGPU() bool
        +getDataType() DataStructureType
        +getMethod() InterpolationMethod
    }
    
    class HermiteMLSInterpolator {
        -coordinates: vector~Point3D~
        -field_data: vector~MagneticFieldData~
        -kd_tree: unique_ptr~KDTree~
        -params: Parameters
        +query(Point3D) InterpolationResult
        +queryBatch(vector) vector
        -computeWeight(Real, Real) Real
        -buildLeastSquaresSystem() void
        -solveLeastSquares() MagneticFieldData
    }
    
    class CPUHermiteMLSInterpolatorAdapter {
        -hmls_interpolator: unique_ptr
        -method: InterpolationMethod
        -extrapolation: ExtrapolationMethod
        +query(Point3D) InterpolationResult
        +queryBatch(vector) vector
        +supportsGPU() bool
    }
    
    class GPUHermiteMLSInterpolatorAdapter {
        -hmls_interpolator: unique_ptr
        -d_points: GpuMemory~Point3D~
        -d_field_data: GpuMemory~MagneticFieldData~
        -spatial_grid: SpatialGrid
        +query(Point3D) InterpolationResult
        +queryBatch(vector) vector
        +supportsGPU() bool
    }
    
    IInterpolator <|-- CPUHermiteMLSInterpolatorAdapter
    IInterpolator <|-- GPUHermiteMLSInterpolatorAdapter
    CPUHermiteMLSInterpolatorAdapter o-- HermiteMLSInterpolator
    GPUHermiteMLSInterpolatorAdapter o-- HermiteMLSInterpolator
```

## Weighted Least Squares System Structure

For a quadratic basis with k neighbors:

```
System Matrix A (size: 4k x 10):
┌─────────────────────────────────┐
│ Value Constraints (k rows)      │  w_1^0.5 * [1, x_1-q_x, y_1-q_y, z_1-q_z, (x_1-q_x)^2, ...]
│                                 │  w_2^0.5 * [1, x_2-q_x, y_2-q_y, z_2-q_z, (x_2-q_x)^2, ...]
│                                 │  ...
├─────────────────────────────────┤
│ X-Derivative Constraints (k)    │  λ^0.5 * w_1^0.5 * d/dx[basis]
│                                 │  λ^0.5 * w_2^0.5 * d/dx[basis]
├─────────────────────────────────┤
│ Y-Derivative Constraints (k)    │  λ^0.5 * w_1^0.5 * d/dy[basis]
│                                 │  ...
├─────────────────────────────────┤
│ Z-Derivative Constraints (k)    │  λ^0.5 * w_1^0.5 * d/dz[basis]
│                                 │  ...
└─────────────────────────────────┘

Right-hand side b (size: 4k x 1):
┌─────────┐
│ f_1     │  Function values
│ f_2     │
│ ...     │
├─────────┤
│ df_1/dx │  X-derivatives
│ df_2/dx │
├─────────┤
│ df_1/dy │  Y-derivatives
│ ...     │
├─────────┤
│ df_1/dz │  Z-derivatives
│ ...     │
└─────────┘

Solve: A * p = b
where p is the vector of polynomial coefficients
```

## Integration with Existing Components

```mermaid
graph LR
    subgraph "New HMLS Components"
        HMLS[HermiteMLSInterpolator]
        HMLSAdapter[HMLS Adapters]
        HMLSKernel[HMLS CUDA Kernel]
    end
    
    subgraph "Existing Components - Reused"
        KDTree[KD-Tree]
        SpatialGrid[Spatial Grid]
        MemMgr[Memory Manager]
        Types[Types & Enums]
        Factory[Factory System]
    end
    
    HMLS --> KDTree
    HMLSKernel --> SpatialGrid
    HMLSAdapter --> MemMgr
    HMLSAdapter --> Types
    Factory --> HMLSAdapter
    
    style HMLS fill:#90EE90
    style HMLSAdapter fill:#90EE90
    style HMLSKernel fill:#90EE90
```

## Testing Strategy

```mermaid
graph TD
    subgraph "Unit Tests"
        UT1[Constructor & Initialization]
        UT2[Weight Function Tests]
        UT3[Basis Function Tests]
        UT4[System Solving Tests]
        UT5[Single Query Tests]
        UT6[Batch Query Tests]
    end
    
    subgraph "Integration Tests"
        IT1[CPU-GPU Consistency]
        IT2[Factory Integration]
        IT3[API Integration]
    end
    
    subgraph "Accuracy Tests"
        AT1[Polynomial Reproduction]
        AT2[Gradient Accuracy]
        AT3[Comparison with IDW]
        AT4[Cross-validation]
    end
    
    subgraph "Performance Tests"
        PT1[CPU Benchmarks]
        PT2[GPU Benchmarks]
        PT3[Scalability Tests]
        PT4[Memory Usage]
    end
    
    UT1 --> IT1
    UT2 --> IT1
    UT3 --> IT1
    UT4 --> IT1
    UT5 --> IT1
    UT6 --> IT1
    
    IT1 --> AT1
    IT2 --> AT1
    IT3 --> AT1
    
    AT1 --> PT1
    AT2 --> PT1
    AT3 --> PT1
    AT4 --> PT1
```
