"""Schedule manager for ETL pipelines - handles cron job registration and management"""

import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
from croniter import croniter
from datetime import datetime

logger = logging.getLogger(__name__)


class ScheduleManager:
    """Manages ETL pipeline schedules"""
    
    def __init__(self, schedules_dir: Path):
        self.schedules_dir = Path(schedules_dir)
        self.schedules = self._load_all_schedules()
        
    def _load_all_schedules(self) -> Dict:
        """Load all schedule YAML files"""
        schedules = {}
        
        for schedule_file in self.schedules_dir.glob("*.yaml"):
            with open(schedule_file, 'r') as f:
                schedule_name = schedule_file.stem
                schedules[schedule_name] = yaml.safe_load(f)
                logger.info(f"Loaded schedule: {schedule_name}")
                
        return schedules
        
    def get_schedule(self, name: str) -> Optional[Dict]:
        """Get a specific schedule by name"""
        return self.schedules.get(name)
        
    def get_next_run_time(self, schedule_name: str) -> Optional[datetime]:
        """Get the next scheduled run time for a schedule"""
        schedule = self.get_schedule(schedule_name)
        if not schedule:
            return None
            
        cron_expr = schedule.get('schedule', {}).get('cron')
        if not cron_expr:
            return None
            
        cron = croniter(cron_expr, datetime.now())
        return cron.get_next(datetime)
        
    def should_run_now(self, schedule_name: str, last_run: datetime) -> bool:
        """Check if a schedule should run now based on last run time"""
        next_run = self.get_next_run_time(schedule_name)
        if not next_run:
            return False
            
        return datetime.now() >= next_run
        
    def validate_schedule(self, schedule_name: str) -> bool:
        """Validate a schedule configuration"""
        schedule = self.get_schedule(schedule_name)
        if not schedule:
            logger.error(f"Schedule {schedule_name} not found")
            return False
            
        # Validate cron expression
        cron_expr = schedule.get('schedule', {}).get('cron')
        if not cron_expr:
            logger.error(f"No cron expression in {schedule_name}")
            return False
            
        try:
            croniter(cron_expr)
        except Exception as e:
            logger.error(f"Invalid cron expression: {e}")
            return False
            
        # Validate parameters
        parameters = schedule.get('parameters', {})
        if not isinstance(parameters, dict):
            logger.error("Parameters must be a dictionary")
            return False
            
        logger.info(f"Schedule {schedule_name} is valid")
        return True
        
    def list_schedules(self) -> Dict[str, Dict]:
        """List all schedules with their next run times"""
        result = {}
        
        for name, schedule in self.schedules.items():
            next_run = self.get_next_run_time(name)
            result[name] = {
                'description': schedule.get('schedule', {}).get('description', ''),
                'cron': schedule.get('schedule', {}).get('cron', ''),
                'next_run': next_run.isoformat() if next_run else None,
                'enabled': schedule.get('enabled', True)
            }
            
        return result


def register_schedules_with_prefect():
    """Register schedules with Prefect deployment"""
    # This function would integrate with Prefect Cloud/Server
    # to register the schedules as Prefect deployments
    
    manager = ScheduleManager(Path(__file__).parent)
    
    for schedule_name, schedule in manager.schedules.items():
        cron = schedule.get('schedule', {}).get('cron')
        flow_name = schedule.get('schedule', {}).get('flow_name', schedule_name)
        
        logger.info(f"Registering schedule: {schedule_name} -> {flow_name} ({cron})")
        
        # Example Prefect registration:
        # from prefect.deployments import Deployment
        # from prefect.infrastructure import KubernetesJob
        # 
        # deployment = Deployment.build_from_flow(
        #     flow=flow,
        #     name=schedule_name,
        #     schedule=cron,
        #     work_pool_name="trading-ai-pool",
        # )
        # deployment.apply()


if __name__ == "__main__":
    # Test the schedules
    manager = ScheduleManager(Path(__file__).parent)
    
    print("=" * 60)
    print("ETL Schedules")
    print("=" * 60)
    
    for name, info in manager.list_schedules().items():
        print(f"\n📅 {name}")
        print(f"   Description: {info['description']}")
        print(f"   Cron: {info['cron']}")
        print(f"   Next Run: {info['next_run']}")
        
    print("\n" + "=" * 60)
    
    # Validate all schedules
    all_valid = True
    for name in manager.schedules.keys():
        if not manager.validate_schedule(name):
            all_valid = False
            print(f"❌ {name}: INVALID")
        else:
            print(f"✅ {name}: VALID")
            
    print("=" * 60)
