"""
================================================================================
GTA 5 PLAYER BEHAVIOR & ANALYTICS PLATFORM - COMPLETE UNIFIED SYSTEM
================================================================================

Production-ready analytics engine for game telemetry, player segmentation,
and business insights. This file contains the entire system in one place.

Author: Senior Fullstack Developer & Lead Game Data Scientist
Version: 1.0.0
License: MIT

BUSINESS IMPACT:
---------------
1. Churn Prevention: Identify at-risk players before they leave (saves ~$40/user CAC)
2. Revenue Optimization: Target high-LTV players with personalized Shark Card offers
3. Game Balance: Detect "death zones" that frustrate players and reduce engagement
4. Content Strategy: Understand which mission types drive retention vs. monetization

USAGE:
------
# Run as standalone analytics demo:
python gta5_analytics_complete.py demo

# Run as API server:
python gta5_analytics_complete.py api

# Run tests:
python gta5_analytics_complete.py test

REQUIREMENTS:
------------
pip install fastapi uvicorn pandas numpy scikit-learn pydantic

API ENDPOINTS (when running as server):
---------------------------------------
- GET  /health                      - Health check
- GET  /api/v1/players              - List players with filtering
- GET  /api/v1/players/{player_id}  - Get individual player profile
- GET  /api/v1/insights/churn       - Churn risk analysis
- GET  /api/v1/insights/segments    - Player segmentation insights
- GET  /api/v1/insights/ltv         - LTV rankings
- GET  /api/v1/insights/heatmap     - Spatial death heatmap
- GET  /api/v1/insights/economy     - Economic balance metrics
- GET  /api/v1/insights/executive   - Executive dashboard summary
- POST /api/v1/refresh              - Re-run analytics pipeline

================================================================================
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PlayerMetrics:
    """Data class representing key player metrics."""
    player_id: str
    total_playtime_hours: float
    session_count: int
    avg_session_duration: float
    death_count: int
    mission_completion_rate: float
    shark_card_revenue: float
    in_game_wealth: float
    days_since_last_login: int
    mission_failure_rate: float
    business_income: float
    heist_income: float


# ============================================================================
# TELEMETRY SIMULATION
# ============================================================================

class MockTelemetry:
    """
    Generates realistic GTA 5 player telemetry data for testing and development.
    
    Simulates:
    - Spatial data (coordinates, districts)
    - Economic events (purchases, business income, heist payouts)
    - Engagement metrics (session duration, deaths, mission outcomes)
    
    Business Value: Enables testing of analytics pipelines without production data access.
    """
    
    DISTRICTS = [
        'Los Santos Downtown', 'Vinewood Hills', 'Vespucci Beach', 
        'Sandy Shores', 'Paleto Bay', 'Blaine County', 'Del Perro',
        'Mirror Park', 'La Mesa', 'Cypress Flats'
    ]
    
    MISSION_TYPES = [
        'Contact Mission', 'Heist Setup', 'Heist Finale', 
        'Business Battle', 'Freemode Event', 'Adversary Mode'
    ]
    
    BUSINESSES = [
        'Bunker', 'Nightclub', 'MC Clubhouse', 'CEO Warehouse',
        'Import/Export Garage', 'Arcade', 'Agency'
    ]
    
    @staticmethod
    def generate_player_events(
        num_players: int = 1000,
        days_history: int = 90
    ) -> pd.DataFrame:
        """
        Generate mock player event log data.
        
        Args:
            num_players: Number of unique players to simulate
            days_history: Number of days of historical data to generate
            
        Returns:
            DataFrame with columns: player_id, timestamp, event_type, event_data
        """
        logger.info(f"Generating telemetry for {num_players} players over {days_history} days")
        
        events = []
        np.random.seed(42)
        
        for player_idx in range(num_players):
            player_id = f"P{player_idx:06d}"
            
            # Player archetype affects behavior patterns
            archetype = np.random.choice(['whale', 'grinder', 'casual', 'churned'], 
                                        p=[0.05, 0.25, 0.50, 0.20])
            
            # Generate sessions based on archetype
            if archetype == 'whale':
                sessions_per_week = np.random.randint(10, 20)
                shark_card_prob = 0.6
                avg_session_hours = np.random.uniform(2, 4)
            elif archetype == 'grinder':
                sessions_per_week = np.random.randint(15, 30)
                shark_card_prob = 0.05
                avg_session_hours = np.random.uniform(3, 6)
            elif archetype == 'casual':
                sessions_per_week = np.random.randint(2, 7)
                shark_card_prob = 0.15
                avg_session_hours = np.random.uniform(1, 2)
            else:  # churned
                sessions_per_week = np.random.randint(1, 3)
                shark_card_prob = 0.01
                avg_session_hours = np.random.uniform(0.5, 1)
            
            total_sessions = int((days_history / 7) * sessions_per_week)
            
            # Simulate declining engagement for churned players
            if archetype == 'churned':
                if days_history > 30:
                    last_active_day = np.random.randint(30, days_history)
                elif days_history > 7:
                    last_active_day = np.random.randint(7, days_history)
                else:
                    last_active_day = np.random.randint(max(1, days_history - 2), days_history)
            else:
                last_active_day = np.random.randint(0, max(1, min(7, days_history)))
            
            for session_idx in range(total_sessions):
                # Session timestamp
                days_ago = np.random.randint(last_active_day, days_history)
                session_time = datetime.now() - timedelta(days=days_ago)
                session_duration = np.random.exponential(avg_session_hours)
                
                # Session start event
                events.append({
                    'player_id': player_id,
                    'timestamp': session_time,
                    'event_type': 'session_start',
                    'event_data': {
                        'archetype': archetype,
                        'session_id': f"{player_id}_S{session_idx}"
                    }
                })
                
                # Movement events during session
                num_movements = int(session_duration * 60)  # ~1 per minute
                for _ in range(num_movements):
                    event_time = session_time + timedelta(minutes=np.random.uniform(0, session_duration * 60))
                    district = np.random.choice(MockTelemetry.DISTRICTS)
                    
                    events.append({
                        'player_id': player_id,
                        'timestamp': event_time,
                        'event_type': 'movement',
                        'event_data': {
                            'x': np.random.uniform(-4000, 4000),
                            'y': np.random.uniform(-4000, 4000),
                            'z': np.random.uniform(0, 500),
                            'district': district
                        }
                    })
                
                # Death events (higher for churned/frustrated players)
                death_multiplier = 2.0 if archetype == 'churned' else 1.0
                num_deaths = int(np.random.poisson(session_duration * 3 * death_multiplier))
                for _ in range(num_deaths):
                    event_time = session_time + timedelta(minutes=np.random.uniform(0, session_duration * 60))
                    district = np.random.choice(MockTelemetry.DISTRICTS)
                    
                    events.append({
                        'player_id': player_id,
                        'timestamp': event_time,
                        'event_type': 'wasted',
                        'event_data': {
                            'district': district,
                            'cause': np.random.choice(['Combat', 'Vehicle', 'Fall', 'Explosion'])
                        }
                    })
                
                # Mission events
                num_missions = int(session_duration * 0.5)  # ~1 every 2 hours
                for _ in range(num_missions):
                    event_time = session_time + timedelta(minutes=np.random.uniform(0, session_duration * 60))
                    mission_type = np.random.choice(MockTelemetry.MISSION_TYPES)
                    
                    # Mission success rate varies by archetype
                    if archetype == 'grinder':
                        success = np.random.random() > 0.2
                    elif archetype == 'churned':
                        success = np.random.random() > 0.6  # High failure = frustration
                    else:
                        success = np.random.random() > 0.4
                    
                    payout = np.random.uniform(10000, 500000) if success else 0
                    
                    events.append({
                        'player_id': player_id,
                        'timestamp': event_time,
                        'event_type': 'mission_complete' if success else 'mission_failed',
                        'event_data': {
                            'mission_type': mission_type,
                            'payout': payout
                        }
                    })
                
                # Business income (passive)
                if archetype in ['whale', 'grinder']:
                    num_business_events = int(session_duration * 0.3)
                    for _ in range(num_business_events):
                        event_time = session_time + timedelta(minutes=np.random.uniform(0, session_duration * 60))
                        business = np.random.choice(MockTelemetry.BUSINESSES)
                        income = np.random.uniform(20000, 150000)
                        
                        events.append({
                            'player_id': player_id,
                            'timestamp': event_time,
                            'event_type': 'business_income',
                            'event_data': {
                                'business': business,
                                'amount': income
                            }
                        })
                
                # Shark Card purchases
                if np.random.random() < shark_card_prob:
                    event_time = session_time + timedelta(minutes=np.random.uniform(0, session_duration * 60))
                    card_tiers = {
                        'Megalodon': (99.99, 8000000),
                        'Whale': (49.99, 3500000),
                        'Bull': (29.99, 2000000),
                        'Great White': (19.99, 1250000)
                    }
                    card_name = np.random.choice(list(card_tiers.keys()))
                    price, gta_dollars = card_tiers[card_name]
                    
                    events.append({
                        'player_id': player_id,
                        'timestamp': event_time,
                        'event_type': 'shark_card_purchase',
                        'event_data': {
                            'card_name': card_name,
                            'usd_price': price,
                            'gta_dollars': gta_dollars
                        }
                    })
                
                # Expenditures
                num_purchases = int(session_duration * 0.4)
                for _ in range(num_purchases):
                    event_time = session_time + timedelta(minutes=np.random.uniform(0, session_duration * 60))
                    amount = np.random.uniform(5000, 2000000)
                    
                    events.append({
                        'player_id': player_id,
                        'timestamp': event_time,
                        'event_type': 'expenditure',
                        'event_data': {
                            'category': np.random.choice(['Vehicle', 'Property', 'Weapon', 'Clothing']),
                            'amount': amount
                        }
                    })
                
                # Session end
                end_time = session_time + timedelta(hours=session_duration)
                events.append({
                    'player_id': player_id,
                    'timestamp': end_time,
                    'event_type': 'session_end',
                    'event_data': {
                        'duration_hours': session_duration
                    }
                })
        
        df = pd.DataFrame(events)
        logger.info(f"Generated {len(df)} total events")
        return df


# ============================================================================
# DATA INGESTION & CLEANING
# ============================================================================

class DataIngestor:
    """
    Cleans and validates raw telemetry data for analytics processing.
    
    Business Value: Ensures data quality, preventing bad insights from corrupted data.
    Handles missing values, outliers, and duplicate events.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        logger.info("DataIngestor initialized")
    
    def clean_telemetry(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate raw telemetry data.
        
        Args:
            raw_df: Raw event DataFrame
            
        Returns:
            Cleaned DataFrame with validated data types and no duplicates
        """
        logger.info(f"Cleaning {len(raw_df)} raw events")
        
        df = raw_df.copy()
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['player_id', 'timestamp', 'event_type'])
        logger.info(f"Removed {initial_count - len(df)} duplicate events")
        
        # Validate timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['player_id', 'timestamp'])
        
        # Remove future events (data quality issue)
        df = df[df['timestamp'] <= datetime.now()]
        
        return df
    
    def aggregate_player_metrics(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate event-level data into player-level metrics.
        
        Args:
            events_df: Cleaned event DataFrame
            
        Returns:
            DataFrame with one row per player and aggregated metrics
        """
        logger.info("Aggregating player-level metrics")
        
        players = []
        
        for player_id in events_df['player_id'].unique():
            player_events = events_df[events_df['player_id'] == player_id]
            
            # Session metrics
            sessions = player_events[player_events['event_type'] == 'session_start']
            session_ends = player_events[player_events['event_type'] == 'session_end']
            
            total_sessions = len(sessions)
            total_playtime = session_ends['event_data'].apply(
                lambda x: x.get('duration_hours', 0) if isinstance(x, dict) else 0
            ).sum()
            avg_session_duration = total_playtime / total_sessions if total_sessions > 0 else 0
            
            # Death metrics
            deaths = player_events[player_events['event_type'] == 'wasted']
            death_count = len(deaths)
            
            # Mission metrics
            missions_completed = len(player_events[player_events['event_type'] == 'mission_complete'])
            missions_failed = len(player_events[player_events['event_type'] == 'mission_failed'])
            total_missions = missions_completed + missions_failed
            
            mission_completion_rate = (
                missions_completed / total_missions if total_missions > 0 else 0
            )
            mission_failure_rate = (
                missions_failed / total_missions if total_missions > 0 else 0
            )
            
            # Economic metrics
            shark_cards = player_events[player_events['event_type'] == 'shark_card_purchase']
            shark_card_revenue = shark_cards['event_data'].apply(
                lambda x: x.get('usd_price', 0) if isinstance(x, dict) else 0
            ).sum()
            
            business_events = player_events[player_events['event_type'] == 'business_income']
            business_income = business_events['event_data'].apply(
                lambda x: x.get('amount', 0) if isinstance(x, dict) else 0
            ).sum()
            
            mission_payouts = player_events[
                player_events['event_type'] == 'mission_complete'
            ]['event_data'].apply(
                lambda x: x.get('payout', 0) if isinstance(x, dict) else 0
            ).sum()
            
            expenditures = player_events[player_events['event_type'] == 'expenditure']
            total_spent = expenditures['event_data'].apply(
                lambda x: x.get('amount', 0) if isinstance(x, dict) else 0
            ).sum()
            
            shark_card_gta_dollars = shark_cards['event_data'].apply(
                lambda x: x.get('gta_dollars', 0) if isinstance(x, dict) else 0
            ).sum()
            
            in_game_wealth = (
                business_income + mission_payouts + shark_card_gta_dollars - total_spent
            )
            
            # Recency
            last_login = player_events['timestamp'].max()
            days_since_last_login = (datetime.now() - last_login).days
            
            players.append(PlayerMetrics(
                player_id=player_id,
                total_playtime_hours=total_playtime,
                session_count=total_sessions,
                avg_session_duration=avg_session_duration,
                death_count=death_count,
                mission_completion_rate=mission_completion_rate,
                shark_card_revenue=shark_card_revenue,
                in_game_wealth=in_game_wealth,
                days_since_last_login=days_since_last_login,
                mission_failure_rate=mission_failure_rate,
                business_income=business_income,
                heist_income=mission_payouts
            ))
        
        df = pd.DataFrame([vars(p) for p in players])
        logger.info(f"Aggregated metrics for {len(df)} players")
        return df


# ============================================================================
# BEHAVIOR ANALYSIS & ML
# ============================================================================

class BehaviorAnalyzer:
    """
    ML-driven player segmentation and behavior pattern analysis.
    
    Business Value: Enables personalized marketing, content recommendations,
    and targeted retention campaigns based on player archetypes.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.churn_model = None
        logger.info("BehaviorAnalyzer initialized")
    
    def segment_players(
        self, 
        player_df: pd.DataFrame, 
        n_clusters: int = 4
    ) -> pd.DataFrame:
        """
        Segment players using K-Means clustering.
        
        Segments typically emerge as:
        - Whales: High spend, low playtime
        - Grinders: Low spend, high playtime, high mission completion
        - Casuals: Medium spend, low playtime
        - At-Risk: Declining engagement, high failure rates
        
        Args:
            player_df: Player metrics DataFrame
            n_clusters: Number of player segments to create
            
        Returns:
            DataFrame with 'segment' column added
        """
        logger.info(f"Segmenting {len(player_df)} players into {n_clusters} clusters")
        
        # Select features for clustering
        feature_cols = [
            'total_playtime_hours',
            'avg_session_duration',
            'mission_completion_rate',
            'shark_card_revenue',
            'in_game_wealth',
            'death_count'
        ]
        
        X = player_df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # K-Means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        player_df['segment'] = self.kmeans_model.fit_predict(X_scaled)
        
        # Label segments based on characteristics
        segment_profiles = player_df.groupby('segment').agg({
            'shark_card_revenue': 'mean',
            'total_playtime_hours': 'mean',
            'mission_completion_rate': 'mean'
        })
        
        segment_labels = {}
        for seg_id, row in segment_profiles.iterrows():
            if row['shark_card_revenue'] > segment_profiles['shark_card_revenue'].median():
                if row['total_playtime_hours'] < segment_profiles['total_playtime_hours'].median():
                    segment_labels[seg_id] = 'Whale'
                else:
                    segment_labels[seg_id] = 'Engaged Spender'
            else:
                if row['total_playtime_hours'] > segment_profiles['total_playtime_hours'].median():
                    segment_labels[seg_id] = 'Grinder'
                else:
                    segment_labels[seg_id] = 'Casual'
        
        player_df['segment_label'] = player_df['segment'].map(segment_labels)
        
        logger.info("Segmentation complete:")
        for label, count in player_df['segment_label'].value_counts().items():
            logger.info(f"  {label}: {count} players ({count/len(player_df)*100:.1f}%)")
        
        return player_df
    
    def predict_churn_risk(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn risk using engagement velocity and failure patterns.
        
        High-risk indicators:
        - Increasing days_since_last_login
        - Rising mission_failure_rate
        - Declining session_duration
        
        Args:
            player_df: Player metrics DataFrame
            
        Returns:
            DataFrame with 'churn_risk' and 'churn_score' columns
        """
        logger.info("Calculating churn risk scores")
        
        # Define churn based on inactivity
        player_df['is_churned'] = (player_df['days_since_last_login'] > 14).astype(int)
        
        # Feature engineering for churn prediction
        player_df['death_per_hour'] = (
            player_df['death_count'] / player_df['total_playtime_hours']
        ).fillna(0)
        
        player_df['engagement_score'] = (
            player_df['mission_completion_rate'] * player_df['avg_session_duration']
        )
        
        # Simple rule-based churn scoring (in production, use trained model)
        player_df['churn_score'] = (
            (player_df['days_since_last_login'] / 30) * 0.4 +
            (player_df['mission_failure_rate']) * 0.3 +
            (1 - player_df['mission_completion_rate']) * 0.3
        )
        
        # Categorize risk levels
        player_df['churn_risk'] = pd.cut(
            player_df['churn_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        risk_distribution = player_df['churn_risk'].value_counts()
        logger.info("Churn risk distribution:")
        for risk, count in risk_distribution.items():
            logger.info(f"  {risk}: {count} players ({count/len(player_df)*100:.1f}%)")
        
        return player_df
    
    def calculate_ltv(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate player Lifetime Value (LTV) based on spending patterns.
        
        LTV Model:
        - Historical shark card revenue
        - Correlation between in-game wealth and purchase probability
        - Session frequency as engagement proxy
        
        Args:
            player_df: Player metrics DataFrame
            
        Returns:
            DataFrame with 'predicted_ltv' column
        """
        logger.info("Calculating player LTV")
        
        # LTV = Historical Revenue + Predicted Future Revenue
        player_df['purchase_probability'] = (
            player_df['shark_card_revenue'] > 0
        ).astype(float)
        
        # Players with low in-game wealth but high engagement are likely future buyers
        player_df['wealth_deficit'] = np.maximum(
            0, 
            player_df['in_game_wealth'].median() - player_df['in_game_wealth']
        )
        
        player_df['future_revenue_estimate'] = (
            player_df['purchase_probability'] * 
            (player_df['wealth_deficit'] / 1000000) *  # Scale to reasonable amounts
            (player_df['session_count'] / 30) *  # Monthly session factor
            25  # Average purchase amount
        )
        
        player_df['predicted_ltv'] = (
            player_df['shark_card_revenue'] + 
            player_df['future_revenue_estimate']
        )
        
        ltv_stats = player_df['predicted_ltv'].describe()
        logger.info(f"LTV Statistics: Mean=${ltv_stats['mean']:.2f}, Median=${ltv_stats['50%']:.2f}")
        
        return player_df


# ============================================================================
# INSIGHT GENERATION
# ============================================================================

class InsightGenerator:
    """
    Generates business KPIs and actionable insights from analyzed player data.
    
    Business Value: Translates raw metrics into executive dashboards and
    decision-support tools for game designers and stakeholders.
    """
    
    def __init__(self):
        logger.info("InsightGenerator initialized")
    
    def generate_spatial_heatmap(
        self, 
        events_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Aggregate death events by district to identify game balance issues.
        
        Business Impact: Highlights "frustration zones" where players die frequently.
        
        Args:
            events_df: Event-level DataFrame
            
        Returns:
            Dictionary with district-level death statistics
        """
        logger.info("Generating spatial death heatmap")
        
        death_events = events_df[events_df['event_type'] == 'wasted'].copy()
        
        death_events['district'] = death_events['event_data'].apply(
            lambda x: x.get('district', 'Unknown') if isinstance(x, dict) else 'Unknown'
        )
        
        heatmap = death_events.groupby('district').size().sort_values(ascending=False)
        
        # Calculate session exits per district
        session_ends = events_df[events_df['event_type'] == 'session_end']
        
        # Approximate "rage quit" locations (death followed by session end within 5 min)
        rage_quits = []
        for player_id in death_events['player_id'].unique():
            player_deaths = death_events[death_events['player_id'] == player_id]
            player_ends = session_ends[session_ends['player_id'] == player_id]
            
            for _, death in player_deaths.iterrows():
                # Check if session ended within 5 minutes
                nearby_ends = player_ends[
                    (player_ends['timestamp'] > death['timestamp']) &
                    (player_ends['timestamp'] <= death['timestamp'] + timedelta(minutes=5))
                ]
                if len(nearby_ends) > 0:
                    rage_quits.append(death['district'])
        
        rage_quit_map = pd.Series(rage_quits).value_counts()
        
        return {
            'death_heatmap': heatmap.to_dict(),
            'rage_quit_locations': rage_quit_map.to_dict(),
            'total_deaths': len(death_events),
            'analysis': {
                'hottest_zone': heatmap.index[0] if len(heatmap) > 0 else 'N/A',
                'hottest_zone_deaths': int(heatmap.iloc[0]) if len(heatmap) > 0 else 0
            }
        }
    
    def calculate_economic_balance(
        self, 
        player_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze the in-game economy's sink-to-source ratio.
        
        Business Impact: Detects inflation or deflation that affects Shark Card sales.
        
        Args:
            player_df: Player-level metrics
            
        Returns:
            Economic health indicators
        """
        logger.info("Calculating economic balance metrics")
        
        # Aggregate income sources
        total_business_income = player_df['business_income'].sum()
        total_heist_income = player_df['heist_income'].sum()
        total_earned = total_business_income + total_heist_income
        
        # Estimate expenditures
        total_wealth = player_df['in_game_wealth'].sum()
        total_shark_card_gta = player_df['shark_card_revenue'].sum() * 100000
        
        # Sink rate
        estimated_sinks = total_earned + total_shark_card_gta - total_wealth
        
        sink_source_ratio = estimated_sinks / total_earned if total_earned > 0 else 0
        
        health_status = 'Healthy' if 0.7 <= sink_source_ratio <= 1.0 else (
            'Inflation Risk' if sink_source_ratio < 0.7 else 'Data Anomaly'
        )
        
        return {
            'total_earned_gta': total_earned,
            'total_wealth_gta': total_wealth,
            'estimated_sinks_gta': estimated_sinks,
            'sink_source_ratio': sink_source_ratio,
            'economic_health': health_status,
            'recommendation': (
                'Introduce limited-time luxury items' if health_status == 'Inflation Risk'
                else 'Economy balanced'
            )
        }
    
    def identify_content_gaps(
        self, 
        events_df: pd.DataFrame,
        player_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Identify which player segments are underserved by current content.
        
        Args:
            events_df: Event-level data
            player_df: Player-level data with segments
            
        Returns:
            Content gap analysis by segment
        """
        logger.info("Analyzing content gaps by player segment")
        
        # Analyze mission diversity
        mission_events = events_df[
            events_df['event_type'].isin(['mission_complete', 'mission_failed'])
        ].copy()
        
        mission_events['mission_type'] = mission_events['event_data'].apply(
            lambda x: x.get('mission_type', 'Unknown') if isinstance(x, dict) else 'Unknown'
        )
        
        # Join with player segments
        mission_with_segment = mission_events.merge(
            player_df[['player_id', 'segment_label']],
            on='player_id',
            how='left'
        )
        
        # Content consumption by segment
        content_consumption = mission_with_segment.groupby(
            ['segment_label', 'mission_type']
        ).size().unstack(fill_value=0)
        
        # Identify underserved segments
        gaps = {}
        for segment in content_consumption.index:
            segment_total = content_consumption.loc[segment].sum()
            segment_diversity = (content_consumption.loc[segment] > 0).sum()
            
            avg_diversity = content_consumption.apply(lambda x: (x > 0).sum(), axis=1).mean()
            
            if segment_diversity < avg_diversity:
                gaps[segment] = {
                    'total_missions': int(segment_total),
                    'content_types_used': int(segment_diversity),
                    'recommendation': f'Expand content variety for {segment} players'
                }
        
        return {
            'content_consumption_matrix': content_consumption.to_dict(),
            'identified_gaps': gaps,
            'overall_content_diversity': len(mission_events['mission_type'].unique())
        }
    
    def generate_executive_summary(
        self, 
        player_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Create high-level executive dashboard metrics.
        
        Args:
            player_df: Player-level data
            events_df: Event-level data
            
        Returns:
            KPI summary for executive reporting
        """
        logger.info("Generating executive summary")
        
        total_players = len(player_df)
        active_players = len(player_df[player_df['days_since_last_login'] <= 7])
        total_revenue = player_df['shark_card_revenue'].sum()
        avg_ltv = player_df['predicted_ltv'].mean()
        
        high_risk_count = len(player_df[player_df['churn_risk'] == 'High'])
        
        # Revenue by segment
        revenue_by_segment = player_df.groupby('segment_label')['shark_card_revenue'].sum()
        top_segment = revenue_by_segment.idxmax()
        
        return {
            'kpis': {
                'total_players': total_players,
                'weekly_active_users': active_players,
                'wau_percentage': (active_players / total_players * 100) if total_players > 0 else 0,
                'total_revenue_usd': total_revenue,
                'average_ltv_usd': avg_ltv,
                'high_churn_risk_players': high_risk_count,
                'churn_risk_percentage': (high_risk_count / total_players * 100) if total_players > 0 else 0
            },
            'revenue_insights': {
                'top_revenue_segment': top_segment,
                'top_segment_revenue': revenue_by_segment[top_segment],
                'revenue_by_segment': revenue_by_segment.to_dict()
            },
            'action_items': [
                f'{high_risk_count} players at high churn risk - launch retention campaign',
                f'{top_segment} segment drives most revenue - prioritize their content requests',
                'Review spatial heatmap for game balance issues'
            ]
        }


# ============================================================================
# FASTAPI REST API
# ============================================================================

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. API server mode disabled.")


if FASTAPI_AVAILABLE:
    
    # Pydantic models
    class PlayerProfile(BaseModel):
        player_id: str
        total_playtime_hours: float
        session_count: int
        avg_session_duration: float
        death_count: int
        mission_completion_rate: float
        shark_card_revenue: float
        in_game_wealth: float
        days_since_last_login: int
        segment_label: str
        churn_risk: str
        churn_score: float
        predicted_ltv: float
    
    
    class ChurnInsight(BaseModel):
        player_id: str
        churn_risk: str
        churn_score: float
        days_since_last_login: int
        mission_failure_rate: float
        recommendation: str
    
    
    class SegmentInsight(BaseModel):
        segment_label: str
        player_count: int
        avg_revenue: float
        avg_playtime: float
        avg_ltv: float
        characteristics: str
    
    
    class LTVRanking(BaseModel):
        player_id: str
        predicted_ltv: float
        shark_card_revenue: float
        segment_label: str
    
    
    class HeatmapData(BaseModel):
        district: str
        death_count: int
        rage_quit_count: int
    
    
    class HealthResponse(BaseModel):
        status: str
        timestamp: datetime
        players_loaded: int
        events_loaded: int
        last_updated: Optional[datetime]
    
    
    # Global state
    class AnalyticsState:
        def __init__(self):
            self.events_df: Optional[pd.DataFrame] = None
            self.player_df: Optional[pd.DataFrame] = None
            self.last_updated: Optional[datetime] = None
            self.ingestor = DataIngestor()
            self.analyzer = BehaviorAnalyzer()
            self.insight_gen = InsightGenerator()
            self._initialize_sample_data()
        
        def _initialize_sample_data(self):
            logger.info("Initializing with sample dataset...")
            telemetry = MockTelemetry()
            self.events_df = telemetry.generate_player_events(num_players=100, days_history=30)
            self.events_df = self.ingestor.clean_telemetry(self.events_df)
            self.player_df = self.ingestor.aggregate_player_metrics(self.events_df)
            self.player_df = self.analyzer.segment_players(self.player_df)
            self.player_df = self.analyzer.predict_churn_risk(self.player_df)
            self.player_df = self.analyzer.calculate_ltv(self.player_df)
            self.last_updated = datetime.now()
            logger.info(f"Sample data loaded: {len(self.player_df)} players, {len(self.events_df)} events")
    
    
    state = AnalyticsState()
    app = FastAPI(
        title="GTA 5 Player Analytics API",
        description="Production analytics platform for player behavior insights",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            players_loaded=len(state.player_df) if state.player_df is not None else 0,
            events_loaded=len(state.events_df) if state.events_df is not None else 0,
            last_updated=state.last_updated
        )
    
    
    @app.get("/api/v1/players", response_model=List[PlayerProfile])
    async def get_players(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        segment: Optional[str] = None,
        churn_risk: Optional[str] = None
    ):
        df = state.player_df.copy()
        if segment:
            df = df[df['segment_label'] == segment]
        if churn_risk:
            df = df[df['churn_risk'] == churn_risk]
        df = df.iloc[offset:offset + limit]
        return [PlayerProfile(**p) for p in df.to_dict('records')]
    
    
    @app.get("/api/v1/players/{player_id}", response_model=PlayerProfile)
    async def get_player(player_id: str):
        player_data = state.player_df[state.player_df['player_id'] == player_id]
        if len(player_data) == 0:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        return PlayerProfile(**player_data.iloc[0].to_dict())
    
    
    @app.get("/api/v1/insights/churn", response_model=List[ChurnInsight])
    async def get_churn_insights(
        risk_level: str = Query("High", regex="^(Low|Medium|High)$"),
        limit: int = Query(50, ge=1, le=500)
    ):
        df = state.player_df[state.player_df['churn_risk'] == risk_level].copy()
        df = df.nlargest(limit, 'churn_score')
        
        insights = []
        for _, row in df.iterrows():
            if row['mission_failure_rate'] > 0.5:
                recommendation = "Offer mission assistance or difficulty adjustment"
            elif row['days_since_last_login'] > 7:
                recommendation = "Send re-engagement email with bonus offer"
            elif row['shark_card_revenue'] > 0:
                recommendation = "VIP retention: Personal outreach + exclusive content"
            else:
                recommendation = "General retention campaign: Double GTA$ weekend"
            
            insights.append(ChurnInsight(
                player_id=row['player_id'],
                churn_risk=row['churn_risk'],
                churn_score=row['churn_score'],
                days_since_last_login=row['days_since_last_login'],
                mission_failure_rate=row['mission_failure_rate'],
                recommendation=recommendation
            ))
        return insights
    
    
    @app.get("/api/v1/insights/segments", response_model=List[SegmentInsight])
    async def get_segment_insights():
        segments = state.player_df.groupby('segment_label').agg({
            'player_id': 'count',
            'shark_card_revenue': 'mean',
            'total_playtime_hours': 'mean',
            'predicted_ltv': 'mean'
        }).reset_index()
        
        insights = []
        for _, row in segments.iterrows():
            if row['segment_label'] == 'Whale':
                characteristics = "High-value spenders, prioritize luxury content"
            elif row['segment_label'] == 'Grinder':
                characteristics = "Low-spend but highly engaged, need challenging content"
            elif row['segment_label'] == 'Casual':
                characteristics = "Moderate engagement, benefit from guided experiences"
            else:
                characteristics = "Mixed profile, requires further analysis"
            
            insights.append(SegmentInsight(
                segment_label=row['segment_label'],
                player_count=int(row['player_id']),
                avg_revenue=float(row['shark_card_revenue']),
                avg_playtime=float(row['total_playtime_hours']),
                avg_ltv=float(row['predicted_ltv']),
                characteristics=characteristics
            ))
        return insights
    
    
    @app.get("/api/v1/insights/ltv", response_model=List[LTVRanking])
    async def get_ltv_rankings(limit: int = Query(100, ge=1, le=500)):
        df = state.player_df.nlargest(limit, 'predicted_ltv')
        return [LTVRanking(
            player_id=row['player_id'],
            predicted_ltv=row['predicted_ltv'],
            shark_card_revenue=row['shark_card_revenue'],
            segment_label=row['segment_label']
        ) for _, row in df.iterrows()]
    
    
    @app.get("/api/v1/insights/heatmap", response_model=List[HeatmapData])
    async def get_spatial_heatmap():
        heatmap_data = state.insight_gen.generate_spatial_heatmap(state.events_df)
        death_map = heatmap_data['death_heatmap']
        rage_quit_map = heatmap_data['rage_quit_locations']
        
        heatmap = [HeatmapData(
            district=district,
            death_count=death_count,
            rage_quit_count=rage_quit_map.get(district, 0)
        ) for district, death_count in death_map.items()]
        heatmap.sort(key=lambda x: x.death_count, reverse=True)
        return heatmap
    
    
    @app.get("/api/v1/insights/economy")
    async def get_economic_balance():
        return state.insight_gen.calculate_economic_balance(state.player_df)
    
    
    @app.get("/api/v1/insights/executive")
    async def get_executive_summary():
        return state.insight_gen.generate_executive_summary(state.player_df, state.events_df)


# ============================================================================
# CLI & DEMO FUNCTIONS
# ============================================================================

def run_demo():
    """Run standalone analytics demo."""
    print("\n" + "="*80)
    print("GTA 5 PLAYER ANALYTICS ENGINE - DEMO MODE")
    print("="*80 + "\n")
    
    # Generate sample data
    print("📊 Generating sample telemetry data...")
    telemetry = MockTelemetry()
    events_df = telemetry.generate_player_events(num_players=100, days_history=30)
    
    # Clean and aggregate
    ingestor = DataIngestor()
    events_clean = ingestor.clean_telemetry(events_df)
    player_df = ingestor.aggregate_player_metrics(events_clean)
    
    # Run analysis
    analyzer = BehaviorAnalyzer()
    player_df = analyzer.segment_players(player_df)
    player_df = analyzer.predict_churn_risk(player_df)
    player_df = analyzer.calculate_ltv(player_df)
    
    # Generate insights
    insight_gen = InsightGenerator()
    summary = insight_gen.generate_executive_summary(player_df, events_clean)
    heatmap = insight_gen.generate_spatial_heatmap(events_clean)
    
    # Display results
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    kpis = summary['kpis']
    print(f"\n📈 Key Metrics:")
    print(f"  • Total Players: {kpis['total_players']:,}")
    print(f"  • Weekly Active: {kpis['weekly_active_users']:,} ({kpis['wau_percentage']:.1f}%)")
    print(f"  • Revenue: ${kpis['total_revenue_usd']:,.2f}")
    print(f"  • Avg LTV: ${kpis['average_ltv_usd']:.2f}")
    print(f"  • High Risk: {kpis['high_churn_risk_players']} ({kpis['churn_risk_percentage']:.1f}%)")
    
    print(f"\n🗺️  Death Heatmap (Top 5):")
    for district, count in list(heatmap['death_heatmap'].items())[:5]:
        print(f"  • {district}: {count:,} deaths")
    
    print("\n✓ Demo complete!\n")


def run_api():
    """Run FastAPI server."""
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    print("\nStarting GTA 5 Analytics API server...")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def run_tests():
    """Run basic validation tests."""
    print("\nRunning tests...\n")
    
    # Test data generation
    telemetry = MockTelemetry()
    events_df = telemetry.generate_player_events(num_players=10, days_history=7)
    assert len(events_df) > 0, "Event generation failed"
    print("✓ Data generation test passed")
    
    # Test ingestion
    ingestor = DataIngestor()
    clean_df = ingestor.clean_telemetry(events_df)
    player_df = ingestor.aggregate_player_metrics(clean_df)
    assert len(player_df) == 10, "Player aggregation failed"
    print("✓ Data ingestion test passed")
    
    # Test analysis
    analyzer = BehaviorAnalyzer()
    player_df = analyzer.segment_players(player_df)
    player_df = analyzer.predict_churn_risk(player_df)
    player_df = analyzer.calculate_ltv(player_df)
    assert 'segment_label' in player_df.columns, "Segmentation failed"
    print("✓ Behavior analysis test passed")
    
    print("\n✓ All tests passed!\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "demo":
            run_demo()
        elif mode == "api":
            run_api()
        elif mode == "test":
            run_tests()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python gta5_analytics_complete.py [demo|api|test]")
            sys.exit(1)
    else:
        # Default: run demo
        run_demo()
