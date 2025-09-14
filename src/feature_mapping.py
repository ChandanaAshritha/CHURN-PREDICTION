FEATURE_MAPPING = {
    'feature_0': 'premium_to_age_ratio',
    'feature_1': 'claim_frequency',
    'feature_2': 'policy_tenure_scaled',
    'feature_3': 'payment_delay_score',
    'feature_4': 'service_interaction_count',
    'feature_5': 'discount_eligibility_score',
    'feature_6': 'risk_score',
    'feature_7': 'region_code',
    'feature_8': 'sales_channel_id',
    'feature_9': 'policy_type',
    'feature_10': 'renewal_status',
    'feature_11': 'family_plan_flag',
    'feature_12': 'auto_renew_flag',
    'feature_13': 'digital_engagement_level',
    'feature_14': 'days_associated',
    'feature_15': 'claim_count_last_year'
}

REVERSE_MAPPING = {v: k for k, v in FEATURE_MAPPING.items()}