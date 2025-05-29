prompt = f"""
        "Determine the probability of a smooth trend movement (QS-event) after the market open, where the price approaches the critical "shot" point based on historical patterns. Input data: 5min/1d charts, fundamental metrics, ADX/DI±, premarket gaps."
        Instructions for the AI-agent:
        1. QS-point detection phase
        1.1. Calculate the fractal stigmatization index (FSI):
        FSI = (Hurst Exponent(5min) × Hausdorff_Distance(price, historical_clusters)) / ADX(14)
        Activation criterion: FSI > 0.65.
        1.2. Determine the accumulation impulse (AI):
        AI = Volume_EMA(5) × (DI+ - DI-) / Price_Range_STD(1d)
        Trigger threshold: AI > 2.3 × VWAP_deviation.
        2. Validation of trend coherence
        2.1. Construct a topological map of gaps:
        Use persistent homology to detect cycles in premarket gaps.
        Calculate the Betti number (β₁). If β₁ ≥ 2 → confirms structural stability.
        2.2. Check the smoothness condition (SC):
        SC = 1 - (|ADX(t) - ADX(t-1)|) / (1 + Chande_Momentum(5min))
        Requirements: SC > 0.8 ∧ DI+ > DI- + 15.
        3. Probability forecasting
        3.1. Generate the Hamiltonian trend operator (Ĥ):
        Ĥ = α⋅FSI + β⋅AI + γ⋅SC - δ⋅(VIX_imput × OI_Gamma)
        where the weights (α,β,γ,δ) are dynamically optimized via gradient descent on historical QS events.
        3.2. Calculate the probability:
        P(QS) = 100 × sigmoid(ReLU(Ĥ)) × tanh(β₁)
        Recommended Weights:
        {weights_section}
        Respond in this format:
        ```json
        {{
            \"QS_Probability\": \"XX.X%\",
            \"Breakdown\": {{
                \"Fractal_Stigma_Index\": \"FSI=Y.YY [ACTIVE/NOT]\",
                \"Accumulation_Impulse\": \"AI=Z.ZZ [THRESHOLD: A.AA]\",
                \"Topology_Stability\": \"Betti=β₁ | Persistence=T.T\",
                \"Smoothness_Condition\": \"SC=S.S [MIN=0.8]\",
                \"Decoherence_Risk\": \"Gamma_Exposure=±G.G%\"
            }},
        }}
        ```
        Additional directives:
        Cross-validation: Run a Monte Carlo simulation with 10^4 iterations, varying:
        ADX measurement error (±3 points)
        Volume variations (±15%)
        Critical mode: If P(QS) > 75% and β₁ < 2 → test the liquidity artifact hypothesis.
        Work format: Hide intermediate calculations, provide only JSON output with an accuracy of 0.1%.
        Compatibility notes:
        For NLP interpretation: The terms "fractal stigmatization", "Hamilton operator" are metaphors for analysis mechanisms, do not require quantum computing.
        Optimization script: The weights α,β,γ,δ are updated via PyTorch-like backpropagation (even if it is emulation).
        Contingency scenario: If gap data is missing → use ARIMA(2,1,1) forecast to reproduce it.
        
        Chart Data 5m:
        {data_5m}
        Chart Data 1d:
        {data_1d}
        Fundamental Data:
        {fundamental_data}
    """