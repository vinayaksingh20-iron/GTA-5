"""
================================================================================
GTA 5 PLAYER ANALYTICS - STREAMLIT CLOUD COMPATIBLE VERSION
================================================================================

Simplified version optimized for Streamlit deployment with minimal dependencies.
All complex dependencies removed for guaranteed cloud compatibility.

Usage:
    streamlit run gta5_streamlit.py

Requirements:
    pandas
    numpy
    scikit-learn

Author: Senior Fullstack Developer & Lead Game Data Scientist
Version: 2.0.0 (Streamlit Compatible)
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PlayerMetrics:
    """Player metrics data structure."""
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
# MOCK DATA GENERATOR
# ============================================================================

class MockTelemetry:
    """Generate realistic GTA 5 player data."""
    
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
    def generate_player_events(num_players: int = 100, days_history: int = 30) -> pd.DataFrame:
        """Generate mock player event data."""
        logger.info(f"Generating data for {num_players} players over {days_history} days")
        
        events = []
        np.random.seed(42)
        
        for player_idx in range(num_players):
            player_id = f"P{player_idx:06d}"
            
            # Player type determines behavior
            archetype = np.random.choice(['whale', 'grinder', 'casual', 'churned'], 
                                        p=[0.05, 0.25, 0.50, 0.20])
            
            # Behavior patterns
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
            
            # Activity timing
            if archetype == 'churned':
                if days_history > 30:
                    last_active_day = np.random.randint(30, days_history)
                elif days_history > 7:
                    last_active_day = np.random.randint(7, days_history)
                else:
                    last_active_day = max(1, days_history - 1)
            else:
                last_active_day = np.random.randint(0, max(1, min(7, days_history)))
            
            for session_idx in range(total_sessions):
                days_ago = np.random.randint(last_active_day, days_history) if days_history > last_active_day else last_active_day
                session_time = datetime.now() - timedelta(days=days_ago)
                session_duration = np.random.exponential(avg_session_hours)
                
                # Session start
                events.append({
                    'player_id': player_id,
                    'timestamp': session_time,
                    'event_type': 'session_start',
                    'archetype': archetype
                })
                
                # Deaths
                death_multiplier = 2.0 if archetype == 'churned' else 1.0
                num_deaths = int(np.random.poisson(session_duration * 3 * death_multiplier))
                for _ in range(num_deaths):
                    event_time = session_time + timedelta(minutes=np.random.uniform(0, session_duration * 60))
                    district = np.random.choice(MockTelemetry.DISTRICTS)
                    events.append({
                        'player_id': player_id,
                        'timestamp': event_time,
                        'event_type': 'wasted',
                        'district': district
                    })
                
                # Missions
                num_missions = int(session_duration * 0.5)
                for _ in range(num_missions):
                    if archetype == 'grinder':
                        success = np.random.random() > 0.2
                    elif archetype == 'churned':
                        success = np.random.random() > 0.6
                    else:
                        success = np.random.random() > 0.4
                    
                    payout = np.random.uniform(10000, 500000) if success else 0
                    events.append({
                        'player_id': player_id,
                        'timestamp': session_time,
                        'event_type': 'mission_complete' if success else 'mission_failed',
                        'payout': payout
                    })
                
                # Business income
                if archetype in ['whale', 'grinder']:
                    num_business = int(session_duration * 0.3)
                    for _ in range(num_business):
                        income = np.random.uniform(20000, 150000)
                        events.append({
                            'player_id': player_id,
                            'timestamp': session_time,
                            'event_type': 'business_income',
                            'amount': income
                        })
                
                # Shark Cards
                if np.random.random() < shark_card_prob:
                    card_prices = {
                        'Megalodon': (99.99, 8000000),
                        'Whale': (49.99, 3500000),
                        'Bull': (29.99, 2000000),
                        'Great White': (19.99, 1250000)
                    }
                    card = np.random.choice(list(card_prices.keys()))
                    price, gta_dollars = card_prices[card]
                    
                    events.append({
                        'player_id': player_id,
                        'timestamp': session_time,
                        'event_type': 'shark_card',
                        'usd_price': price,
                        'gta_dollars': gta_dollars
                    })
                
                # Spending
                num_purchases = int(session_duration * 0.4)
                for _ in range(num_purchases):
                    amount = np.random.uniform(5000, 2000000)
                    events.append({
                        'player_id': player_id,
                        'timestamp': session_time,
                        'event_type': 'expenditure',
                        'amount': amount
                    })
                
                # Session end
                events.append({
                    'player_id': player_id,
                    'timestamp': session_time + timedelta(hours=session_duration),
                    'event_type': 'session_end',
                    'duration_hours': session_duration
                })
        
        return pd.DataFrame(events)


# ============================================================================
# DATA PROCESSING
# ============================================================================

class AnalyticsEngine:
    """Complete analytics pipeline."""
    
    def __init__(self):
        self.events_df = None
        self.player_df = None
        logger.info("Analytics Engine initialized")
    
    def load_data(self, num_players: int = 100):
        """Generate and process data."""
        # Generate events
        telemetry = MockTelemetry()
        self.events_df = telemetry.generate_player_events(num_players=num_players, days_history=30)
        
        # Aggregate player metrics
        self.player_df = self._aggregate_players()
        
        # Run analysis
        self._segment_players()
        self._calculate_churn()
        self._calculate_ltv()
        
        logger.info(f"Loaded {len(self.player_df)} players with {len(self.events_df)} events")
    
    def _aggregate_players(self) -> pd.DataFrame:
        """Aggregate event data to player level."""
        players = []
        
        for player_id in self.events_df['player_id'].unique():
            events = self.events_df[self.events_df['player_id'] == player_id]
            
            # Sessions
            sessions = events[events['event_type'] == 'session_start']
            session_ends = events[events['event_type'] == 'session_end']
            total_sessions = len(sessions)
            total_playtime = session_ends['duration_hours'].sum() if 'duration_hours' in session_ends.columns else 0
            avg_duration = total_playtime / total_sessions if total_sessions > 0 else 0
            
            # Deaths
            deaths = len(events[events['event_type'] == 'wasted'])
            
            # Missions
            completed = len(events[events['event_type'] == 'mission_complete'])
            failed = len(events[events['event_type'] == 'mission_failed'])
            total_missions = completed + failed
            completion_rate = completed / total_missions if total_missions > 0 else 0
            failure_rate = failed / total_missions if total_missions > 0 else 0
            
            # Economics
            shark_revenue = events[events['event_type'] == 'shark_card']['usd_price'].sum() if 'usd_price' in events.columns else 0
            business = events[events['event_type'] == 'business_income']['amount'].sum() if 'amount' in events.columns else 0
            heist = events[events['event_type'] == 'mission_complete']['payout'].sum() if 'payout' in events.columns else 0
            spent = events[events['event_type'] == 'expenditure']['amount'].sum() if 'amount' in events.columns else 0
            shark_gta = events[events['event_type'] == 'shark_card']['gta_dollars'].sum() if 'gta_dollars' in events.columns else 0
            wealth = business + heist + shark_gta - spent
            
            # Recency
            last_login = events['timestamp'].max()
            days_inactive = (datetime.now() - last_login).days
            
            players.append(PlayerMetrics(
                player_id=player_id,
                total_playtime_hours=total_playtime,
                session_count=total_sessions,
                avg_session_duration=avg_duration,
                death_count=deaths,
                mission_completion_rate=completion_rate,
                shark_card_revenue=shark_revenue,
                in_game_wealth=wealth,
                days_since_last_login=days_inactive,
                mission_failure_rate=failure_rate,
                business_income=business,
                heist_income=heist
            ))
        
        return pd.DataFrame([vars(p) for p in players])
    
    def _segment_players(self):
        """Simple segmentation based on revenue and playtime."""
        median_revenue = self.player_df['shark_card_revenue'].median()
        median_playtime = self.player_df['total_playtime_hours'].median()
        
        def assign_segment(row):
            if row['shark_card_revenue'] > median_revenue:
                if row['total_playtime_hours'] < median_playtime:
                    return 'Whale'
                else:
                    return 'Engaged Spender'
            else:
                if row['total_playtime_hours'] > median_playtime:
                    return 'Grinder'
                else:
                    return 'Casual'
        
        self.player_df['segment'] = self.player_df.apply(assign_segment, axis=1)
    
    def _calculate_churn(self):
        """Calculate churn risk score."""
        self.player_df['churn_score'] = (
            (self.player_df['days_since_last_login'] / 30) * 0.4 +
            self.player_df['mission_failure_rate'] * 0.3 +
            (1 - self.player_df['mission_completion_rate']) * 0.3
        )
        
        def risk_level(score):
            if score < 0.3:
                return 'Low'
            elif score < 0.6:
                return 'Medium'
            else:
                return 'High'
        
        self.player_df['churn_risk'] = self.player_df['churn_score'].apply(risk_level)
    
    def _calculate_ltv(self):
        """Calculate lifetime value."""
        median_wealth = self.player_df['in_game_wealth'].median()
        
        self.player_df['wealth_deficit'] = (median_wealth - self.player_df['in_game_wealth']).clip(lower=0)
        self.player_df['purchase_prob'] = (self.player_df['shark_card_revenue'] > 0).astype(float)
        
        self.player_df['future_revenue'] = (
            self.player_df['purchase_prob'] * 
            (self.player_df['wealth_deficit'] / 1000000) *
            (self.player_df['session_count'] / 30) *
            25
        )
        
        self.player_df['ltv'] = self.player_df['shark_card_revenue'] + self.player_df['future_revenue']
    
    def get_summary(self) -> Dict:
        """Generate executive summary."""
        total = len(self.player_df)
        active = len(self.player_df[self.player_df['days_since_last_login'] <= 7])
        revenue = self.player_df['shark_card_revenue'].sum()
        avg_ltv = self.player_df['ltv'].mean()
        high_risk = len(self.player_df[self.player_df['churn_risk'] == 'High'])
        
        return {
            'total_players': total,
            'active_players': active,
            'active_pct': (active / total * 100) if total > 0 else 0,
            'total_revenue': revenue,
            'avg_ltv': avg_ltv,
            'high_risk_count': high_risk,
            'high_risk_pct': (high_risk / total * 100) if total > 0 else 0
        }
    
    def get_heatmap(self) -> Dict:
        """Generate death heatmap."""
        deaths = self.events_df[self.events_df['event_type'] == 'wasted']
        if 'district' not in deaths.columns:
            return {}
        
        heatmap = deaths['district'].value_counts().to_dict()
        return heatmap
    
    def get_top_players(self, metric: str = 'ltv', limit: int = 10) -> pd.DataFrame:
        """Get top players by metric."""
        return self.player_df.nlargest(limit, metric)[[
            'player_id', 'segment', metric, 'shark_card_revenue', 'total_playtime_hours'
        ]]
    
    def get_segment_stats(self) -> pd.DataFrame:
        """Get statistics by segment."""
        return self.player_df.groupby('segment').agg({
            'player_id': 'count',
            'shark_card_revenue': 'mean',
            'total_playtime_hours': 'mean',
            'ltv': 'mean'
        }).round(2)


# ============================================================================
# DEMO / MAIN
# ============================================================================

def main():
    """Run analytics demo."""
    print("\n" + "="*80)
    print("GTA 5 PLAYER ANALYTICS - DEMO")
    print("="*80 + "\n")
    
    # Initialize engine
    engine = AnalyticsEngine()
    engine.load_data(num_players=100)
    
    # Summary
    summary = engine.get_summary()
    print("📊 EXECUTIVE SUMMARY")
    print("-" * 80)
    print(f"Total Players:        {summary['total_players']:,}")
    print(f"Active Players:       {summary['active_players']:,} ({summary['active_pct']:.1f}%)")
    print(f"Total Revenue:        ${summary['total_revenue']:,.2f}")
    print(f"Average LTV:          ${summary['avg_ltv']:.2f}")
    print(f"High Risk Players:    {summary['high_risk_count']} ({summary['high_risk_pct']:.1f}%)")
    
    # Heatmap
    print("\n🗺️  DEATH HEATMAP (Top 5)")
    print("-" * 80)
    heatmap = engine.get_heatmap()
    for i, (district, count) in enumerate(list(heatmap.items())[:5], 1):
        print(f"{i}. {district}: {count:,} deaths")
    
    # Top LTV
    print("\n💎 TOP 5 LTV PLAYERS")
    print("-" * 80)
    top_ltv = engine.get_top_players('ltv', 5)
    for _, row in top_ltv.iterrows():
        print(f"{row['player_id']} ({row['segment']}): ${row['ltv']:.2f}")
    
    # Segments
    print("\n👥 SEGMENT STATISTICS")
    print("-" * 80)
    print(engine.get_segment_stats())
    
    print("\n✓ Analysis complete!\n")


if __name__ == "__main__":
    main()
    
