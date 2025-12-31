---
sidebar_position: 2
---

# Feedback and Closed-Loop Control in Humanoid Robots

## Introduction

Control systems are the foundation of stable, purposeful motion in humanoid robots. These systems continuously monitor the robot's state and adjust control inputs to achieve desired behaviors while maintaining stability. This chapter explores the fundamental principles, architectures, and implementation techniques of control systems in humanoid robotics.

## Control System Fundamentals

### Open-Loop vs. Closed-Loop Control

#### Open-Loop Control

Open-loop control applies predetermined inputs without feedback:

```
Input → System → Output
```

**Characteristics**:
- No feedback from system output
- Simple implementation
- Susceptible to disturbances and model errors
- Suitable for well-characterized systems

**Applications**:
- Pre-programmed motions in predictable environments
- Simple repetitive tasks
- Systems with accurate models

#### Closed-Loop Control

Closed-loop control uses feedback to adjust inputs:

```
Reference → Controller → System → Output
    ↑                     ↓
    └── Feedback ←───────┘
```

**Characteristics**:
- Uses output feedback to adjust inputs
- Robust to disturbances and model errors
- More complex implementation
- Essential for stable humanoid operation

**Advantages**:
- Disturbance rejection
- Model error compensation
- Stability maintenance
- Performance adaptation

### Control System Architecture

#### Single-Input Single-Output (SISO)

SISO systems control one output with one input:

```
R(s) ──→ Σ ──→ Controller ──→ Plant ──→ Y(s)
         ↑                    ↓
         └─────── Feedback ───┘
```

#### Multi-Input Multi-Output (MIMO)

MIMO systems control multiple outputs with multiple inputs:
- **Cross-coupling**: Inputs affect multiple outputs
- **Interaction**: System outputs influence each other
- **Complexity**: Requires advanced control techniques
- **Humanoid relevance**: Multiple joints affect robot state

## Classical Control Techniques

### Proportional-Integral-Derivative (PID) Control

PID control is fundamental in robotics:

#### Mathematical Formulation

```
u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de(t)/dt
```

Where:
- u(t): Control signal
- e(t): Error signal (reference - actual)
- Kp: Proportional gain
- Ki: Integral gain
- Kd: Derivative gain

#### Transfer Function

In Laplace domain:
```
C(s) = Kp + Ki/s + Kd*s = (Kd*s² + Kp*s + Ki)/s
```

#### PID Tuning Methods

##### Ziegler-Nichols Method
- **Step 1**: Set Ki = Kd = 0
- **Step 2**: Increase Kp until sustained oscillations occur
- **Step 3**: Record critical gain Kc and oscillation period Pc
- **Step 4**: Apply tuning rules:
  - P: Kp = 0.5*Kc
  - PI: Kp = 0.45*Kc, Ki = 0.54*Kc/Pc
  - PID: Kp = 0.6*Kc, Ki = 1.2*Kc/Pc, Kd = 0.075*Kc*Pc

##### Cohen-Coon Method
Better for systems with dead time:
- **Process identification**: Determine process gain, time constant, dead time
- **Tuning rules**: Specific formulas based on process parameters
- **Advantages**: Better performance for lag-dominant processes

#### PID Limitations in Humanoids

- **Nonlinearity**: Humanoid dynamics are highly nonlinear
- **Coupling**: Joints interact with each other
- **Constraints**: Actuator limits and safety requirements
- **Disturbances**: External forces and model uncertainties

### Lead-Lag Compensation

Lead-lag compensators modify system frequency response:

#### Lead Compensator
Improves transient response and stability margin:
```
D(s) = (αTs + 1)/(Ts + 1), where α > 1
```

#### Lag Compensator
Improves steady-state accuracy:
```
D(s) = (βTs + 1)/(Ts + 1), where β < 1
```

#### Lead-Lag Compensator
Combines lead and lag effects:
```
D(s) = [(α₁Ts + 1)(β₂Ts + 1)] / [(T₁s + 1)(T₂s + 1)]
```

## Modern Control Techniques

### State-Space Representation

State-space models represent systems with state variables:

#### Linear Time-Invariant (LTI) Systems

```
ẋ(t) = Ax(t) + Bu(t)
y(t) = Cx(t) + Du(t)
```

Where:
- x(t): State vector
- u(t): Input vector
- y(t): Output vector
- A, B, C, D: System matrices

#### Controllability and Observability

**Controllability**: Ability to reach any state from any initial state
```
rank([B, AB, A²B, ..., A^(n-1)B]) = n
```

**Observability**: Ability to determine state from output measurements
```
rank([C, CA, CA², ..., CA^(n-1)])ᵀ = n
```

### Linear Quadratic Regulator (LQR)

LQR provides optimal control for linear systems with quadratic cost:

#### Cost Function

```
J = ∫₀^∞ [xᵀQx + uᵀRu] dt
```

Where Q and R are positive definite weighting matrices.

#### Optimal Control Law

```
u = -Kx
```

Where K = R⁻¹BᵀP and P is solution to Algebraic Riccati Equation:
```
AᵀP + PA - PBR⁻¹BᵀP + Q = 0
```

#### LQR Properties

- **Optimality**: Minimizes quadratic cost function
- **Stability**: Closed-loop system is stable
- **Robustness**: Good gain and phase margins
- **Limitations**: Requires linear system model

### Linear Quadratic Gaussian (LQG) Control

LQG combines LQR with Kalman filtering:

#### Separation Principle

LQG controller consists of:
1. **Kalman Filter**: Estimates system state
2. **LQR Controller**: Applies optimal control

#### System Model

With process and measurement noise:
```
ẋ = Ax + Bu + w
y = Cx + v
```

Where w and v are white noise processes.

## Advanced Control Strategies

### Model Predictive Control (MPC)

MPC solves finite-horizon optimal control problems online:

#### Optimization Problem

At each time step k, solve:
```
min Σ[j=0 to N-1] l(x(k+j), u(k+j)) + V(x(k+N))
```

Subject to:
- System dynamics: x(j+1) = f(x(j), u(j))
- Constraints: x_min ≤ x ≤ x_max, u_min ≤ u ≤ u_max
- Initial condition: x(k) = x_measured

#### MPC Advantages

- **Constraint handling**: Explicitly handles constraints
- **Prediction**: Considers future behavior
- **Optimization**: Minimizes explicit cost function
- **Adaptation**: Updates based on new measurements

#### MPC Implementation Challenges

- **Computational complexity**: Solves optimization online
- **Real-time requirements**: Must solve before next sample
- **Robustness**: Sensitive to model errors
- **Stability**: Requires terminal constraints/costs

### Adaptive Control

Adaptive control adjusts parameters based on system behavior:

#### Model Reference Adaptive Control (MRAC)

System tracks reference model:
```
Plant: ẋ = f(x, u, θ)
Ref. Model: ẋₘ = Aₘxₘ + Bₘr
```

Adjust parameters θ to minimize tracking error e = x - xₘ.

#### Self-Tuning Regulators (STR)

Estimate parameters online and update controller:
1. **Parameter estimation**: Estimate system parameters
2. **Controller design**: Design controller for estimated model
3. **Implementation**: Apply designed controller

#### Direct vs. Indirect Adaptive Control

**Direct**: Adjust controller parameters directly
**Indirect**: Estimate plant parameters, then adjust controller

### Robust Control

Robust control maintains performance despite uncertainties:

#### H∞ Control

Minimizes worst-case gain from disturbances to outputs:
```
||Tzw||∞ = sup_ω σ_max[Tzw(jω)] ≤ γ
```

Where Tzw is transfer function from disturbances to errors.

#### μ-Synthesis

Considers structured uncertainties:
- **Uncertainty modeling**: Represent uncertainties explicitly
- **Synthesis**: Design controller for uncertainty structure
- **Analysis**: Verify robust performance

## Humanoid-Specific Control Challenges

### Balance Control

Balance control maintains stability during motion:

#### Inverted Pendulum Model

Simple model for balance:
```
m*l*θ̈ = m*g*sin(θ) + F_horizontal
```

Where m is mass, l is length, θ is angle, g is gravity, F is horizontal force.

#### Linear Inverted Pendulum Mode (LIPM)

Linearized model:
```
ẍ = ω²(x - x_zmp)
```

Where ω² = g/h, h is CoM height, x_zmp is Zero Moment Point.

#### Capture Point Control

Determine where to step to stop:
```
Capture Point = CoM_position + √(h/g) * CoM_velocity
```

### Whole-Body Control

Whole-body control coordinates all joints simultaneously:

#### Task-Based Control

Define multiple tasks with priorities:
```
min ||J₁q̇ - v₁||² + λ₁||J₂q̇ - v₂||² + λ₂||q̇||²
```

Where J₁, J₂ are task Jacobians, v₁, v₂ are task velocities.

#### Hierarchical Control

Organize tasks in priority hierarchy:
1. **Highest priority**: Balance and safety constraints
2. **High priority**: Primary tasks (walking, manipulation)
3. **Low priority**: Secondary tasks (posture optimization)
4. **Lowest priority**: Joint limit avoidance

#### Operational Space Control

Control in task space while managing null space:
```
τ = J₁ᵀF₁ + (I - J₁ᵀJ₁⁺)J₂ᵀF₂ + τ_null
```

Where J₁⁺ is pseudoinverse of primary task Jacobian.

### Impedance Control

Impedance control regulates interaction forces:

#### Impedance Model

Desired relationship between force and motion:
```
M_d(ẍ_d - ẍ) + B_d(ẋ_d - ẋ) + K_d(x_d - x) = F - F_d
```

Where M_d, B_d, K_d are desired mass, damping, stiffness.

#### Control Implementation

Achieve desired impedance through:
```
τ = τ_ff + JᵀF
```

Where τ_ff is feedforward torque, F is desired force.

### Hybrid Force-Motion Control

Control both position and force simultaneously:

#### Natural Coordinate Framework

Decompose into constrained and unconstrained directions:
- **Constrained**: Force control (contact surfaces)
- **Unconstrained**: Position control (free motion)

#### Orthogonal Decomposition

Separate motion and force control:
```
P_t = I - P_n  (motion subspace)
P_n = J_cᵀ(J_c J_cᵀ)⁻¹ J_c  (force subspace)
```

## Control Architectures

### Hierarchical Control

Control systems organized in layers:

#### High-Level Planner
- **Functions**: Task planning, trajectory generation
- **Rate**: 1-10 Hz
- **Inputs**: Task specifications, environment models
- **Outputs**: Desired trajectories

#### Low-Level Controller
- **Functions**: Joint position/force control
- **Rate**: 100-1000 Hz
- **Inputs**: Desired trajectories, sensor feedback
- **Outputs**: Motor commands

#### Middle-Level Coordinator
- **Functions**: Balance, whole-body coordination
- **Rate**: 50-200 Hz
- **Inputs**: High-level commands, low-level states
- **Outputs**: Coordinated trajectories

### Distributed Control

Control systems distributed across robot:

#### Joint-Level Controllers
- **Location**: At each joint
- **Function**: Local position/velocity control
- **Communication**: Minimal, mostly local feedback

#### Limb-Level Controllers
- **Location**: For each limb
- **Function**: Coordinated limb motion
- **Communication**: With other limbs and central coordinator

#### Central Coordinator
- **Location**: Central processing unit
- **Function**: Overall coordination and balance
- **Communication**: With all subsystems

## Sensor Integration

### State Estimation

Estimate system state from sensor measurements:

#### Extended Kalman Filter (EKF)

Linearize nonlinear system around operating point:
1. **Prediction**: x̂ₖ|ₖ₋₁ = f(x̂ₖ₋₁|ₖ₋₁, uₖ₋₁)
2. **Update**: x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kₖ[yₖ - h(x̂ₖ|ₖ₋₁)]

#### Unscented Kalman Filter (UKF)

Use sigma points to capture distribution:
- **Advantages**: Better for highly nonlinear systems
- **Disadvantages**: More computationally expensive

#### Particle Filters

Represent distribution with particles:
- **Advantages**: Handles multimodal distributions
- **Disadvantages**: High computational cost

### Sensor Fusion

Combine multiple sensor inputs:

#### Complementary Filtering

Combine sensors with different characteristics:
```
State_estimate = α*slow_sensor + (1-α)*fast_sensor
```

Where α is frequency-dependent weight.

#### Kalman Filter Fusion

Optimally combine sensor information:
- **Process model**: Predict state evolution
- **Measurement model**: Relate states to measurements
- **Optimal weights**: Based on sensor accuracies

## Implementation Considerations

### Real-Time Requirements

Control systems must meet timing constraints:

#### Hard Real-Time
- **Deadline**: Missed deadlines are system failures
- **Examples**: Balance control, emergency stops
- **Requirements**: Guaranteed timing performance

#### Soft Real-Time
- **Deadline**: Preferred but not critical
- **Examples**: Trajectory tracking, planning
- **Requirements**: Statistical timing guarantees

#### Anytime Algorithms
- **Property**: Provide best result given available time
- **Applications**: Optimization-based controllers
- **Benefits**: Adapt to computational constraints

### Computational Efficiency

Optimize for limited computational resources:

#### Model Simplification

- **Linearization**: Approximate nonlinear systems
- **Order reduction**: Simplify complex models
- **Look-up tables**: Pre-compute complex functions

#### Numerical Optimization

- **Matrix operations**: Use efficient algorithms
- **Sparsity exploitation**: Take advantage of sparse matrices
- **Parallel processing**: Distribute computations

### Safety and Fault Tolerance

Ensure safe operation despite failures:

#### Safe States

- **Definition**: Robot configurations that prevent harm
- **Examples**: Joint limits, singularity avoidance
- **Implementation**: Constraint enforcement

#### Fault Detection and Isolation

- **Monitoring**: Continuously check system health
- **Detection**: Identify when faults occur
- **Isolation**: Determine fault location and cause

#### Fault-Tolerant Control

- **Redundancy**: Use backup systems
- **Reconfiguration**: Adapt control structure
- **Graceful degradation**: Maintain partial functionality

## Control System Design Process

### System Modeling

Create mathematical model of robot dynamics:

#### Rigid Body Dynamics

Lagrangian formulation:
```
H(q)q̈ + C(q, q̇)q̇ + g(q) = τ + JᵀF_ext
```

Where:
- H(q): Mass/inertia matrix
- C(q, q̇): Coriolis/centrifugal forces
- g(q): Gravitational forces
- τ: Joint torques
- F_ext: External forces

#### Parameter Identification

Determine model parameters:
- **Offline methods**: Systematic experiments
- **Online methods**: Real-time parameter estimation
- **Validation**: Verify model accuracy

### Controller Design

Design control laws based on system model:

#### Performance Specifications

- **Stability**: System remains bounded
- **Accuracy**: Error converges to acceptable level
- **Response time**: Fast enough for application
- **Robustness**: Handles uncertainties

#### Control Structure Selection

Choose appropriate control structure:
- **Feedback linearization**: Cancel nonlinearities
- **Sliding mode**: Robust to uncertainties
- **Adaptive control**: Handle parameter variations

### Stability Analysis

Verify controller stability:

#### Lyapunov Stability

Construct Lyapunov function V(x):
- **Positive definiteness**: V(0) = 0, V(x) > 0 for x ≠ 0
- **Negative definiteness**: V̇(x) < 0 for x ≠ 0

#### Frequency Domain Analysis

Analyze system in frequency domain:
- **Gain margin**: Amount of gain increase before instability
- **Phase margin**: Amount of phase lag before instability
- **Bandwidth**: Frequency range of good performance

### Performance Evaluation

Assess controller performance:

#### Simulation Testing

- **Model verification**: Ensure model accuracy
- **Controller tuning**: Optimize parameters
- **Scenario testing**: Test various conditions

#### Experimental Validation

- **Laboratory testing**: Controlled environment
- **Field testing**: Real-world conditions
- **Iterative refinement**: Improve based on results

## Advanced Control Techniques

### Learning-Based Control

Use machine learning to improve control:

#### Reinforcement Learning

Learn control policies through interaction:
- **State**: Robot and environment state
- **Action**: Control inputs
- **Reward**: Performance measure
- **Policy**: Mapping from state to action

#### Imitation Learning

Learn from expert demonstrations:
- **Behavior cloning**: Direct mapping from state to action
- **Inverse RL**: Learn reward function from demonstrations
- **DAgger**: Dataset Aggregation for learning

#### Model Learning

Learn system dynamics model:
- **System identification**: Parametric model learning
- **Black-box models**: Neural networks for dynamics
- **Uncertainty quantification**: Model confidence

### Event-Based Control

Trigger control updates based on events:

#### Trigger Conditions

- **Error threshold**: Update when error exceeds limit
- **Performance degradation**: Adapt when performance drops
- **Environmental changes**: Respond to new conditions

#### Advantages

- **Efficiency**: Reduce unnecessary updates
- **Resource conservation**: Save computational power
- **Adaptation**: Respond to important events

### Predictive Control

Use predictions for control decisions:

#### Model Predictive Control (MPC)

Optimize over prediction horizon:
- **Prediction model**: Forecast system behavior
- **Optimization**: Minimize predicted cost
- **Receding horizon**: Apply first control, repeat

#### Predictive Filtering

Anticipate future states:
- **State prediction**: Forecast state evolution
- **Uncertainty propagation**: Predict confidence bounds
- **Adaptive sampling**: Adjust based on predictions

## Case Studies

### ASIMO Balance Control

Honda ASIMO's balance control system:
- **ZMP control**: Zero moment point based approach
- **Preview control**: Uses future step information
- **Adaptive gait**: Adjusts to terrain and disturbances
- **Multi-layer architecture**: High-level planning, mid-level coordination, low-level control

### Atlas Dynamic Control

Boston Dynamics Atlas control system:
- **High-frequency control**: 1000 Hz joint control
- **Model predictive control**: Predictive optimization
- **Dynamic recovery**: Automatic recovery from disturbances
- **Force control**: Precise contact force regulation

### NAO Walking Control

Aldebaran NAO walking controller:
- **Pattern-based walking**: Pre-computed walking patterns
- **Feedback control**: Adjust for disturbances
- **Simple but robust**: Reliable for educational use
- **Adaptive parameters**: Adjust for different conditions

## Control Software Frameworks

### ROS Control

Standardized control framework for ROS:
- **Hardware interfaces**: Standardized communication
- **Controller managers**: Runtime controller loading
- **Real-time safety**: Thread-safe execution
- **Plugin architecture**: Extensible controller types

### YARP (Yet Another Robot Platform)

Middleware for robot control:
- **Port-based communication**: Flexible message passing
- **Real-time support**: Priority-based threading
- **Configuration management**: Runtime parameter tuning
- **Cross-platform**: Runs on various hardware

### Drake

Model-based design and verification:
- **Automatic differentiation**: Efficient gradient computation
- **Optimization tools**: Sophisticated optimization algorithms
- **Verification**: Formal verification capabilities
- **Rigorous simulation**: Accurate physical simulation

## Performance Metrics

### Stability Metrics

Quantify control system stability:
- **Lyapunov exponents**: Measure stability margins
- **Gain/phase margins**: Frequency domain stability
- **Region of attraction**: Basin of stable operation

### Performance Metrics

Quantify control system performance:
- **Tracking error**: Deviation from desired trajectory
- **Settling time**: Time to reach steady state
- **Overshoot**: Maximum deviation during response
- **Steady-state error**: Error at convergence

### Robustness Metrics

Quantify system robustness:
- **Stability margins**: Robustness to uncertainties
- **Performance degradation**: Behavior under disturbances
- **Parameter sensitivity**: Response to parameter changes

## Future Directions

### Bio-Inspired Control

Nature-inspired control strategies:
- **Central pattern generators**: Rhythmic motion patterns
- **Muscle-like actuators**: Compliant, bio-mimetic actuation
- **Neuromorphic control**: Brain-inspired processing

### Learning-Based Control

Machine learning for control:
- **Meta-learning**: Learn to learn new tasks quickly
- **Transfer learning**: Apply knowledge to new robots
- **Continual learning**: Learn without forgetting old skills

### Distributed Intelligence

Decentralized control approaches:
- **Swarm intelligence**: Collective behavior emergence
- **Federated learning**: Distributed learning across robots
- **Edge computing**: Localized decision making

## Key Takeaways

- Feedback control is essential for humanoid robot stability
- PID control provides foundation for many applications
- Advanced techniques like MPC offer enhanced performance
- Humanoid-specific challenges require specialized approaches
- Real-time requirements constrain control design
- Safety and fault tolerance are critical considerations
- Simulation and experimental validation are essential
- Software frameworks provide implementation tools

## Looking Forward

The next chapter will explore human-robot interaction systems that enable humanoid robots to communicate and collaborate effectively with humans. We'll examine how control systems support interaction and communication capabilities.