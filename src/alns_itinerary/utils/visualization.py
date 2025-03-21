import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SolutionVisualizer:
    """
    Comprehensive visualization tools for travel itinerary solutions
    """
    @staticmethod
    def create_timeline_plot(problem, solution, filename=None):
        """
        Create a detailed timeline visualization of the solution
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector
            filename: Optional output filename
        
        Returns:
            str: Path to the generated visualization
        """
        try:
            # Evaluate solution to get route details
            evaluation = problem.evaluate_solution(solution)
            
            # Prepare figure
            plt.figure(figsize=(15, 5 * problem.NUM_DAYS))
            plt.suptitle("Travel Itinerary Timeline", fontsize=16)
            
            # Color mapping for location types
            color_map = {
                "hotel": "#808080",      # Gray
                "attraction": "#4CAF50", # Green
                "hawker": "#FF9800"      # Orange
            }
            
            # Create a subplot for each day
            for day_idx, daily_route in enumerate(evaluation.get('daily_routes', [])):
                plt.subplot(problem.NUM_DAYS, 1, day_idx + 1)
                
                # Prepare data for plotting
                names = []
                times = []
                colors = []
                
                # Process each step in the route
                for step in daily_route:
                    names.append(step['name'])
                    times.append(step['time'])
                    colors.append(color_map.get(step['type'], '#1E88E5'))  # Blue as default
                
                # Convert times to datetime for better visualization
                base_date = datetime(2025, 1, 1)  # Arbitrary base date
                datetime_times = [base_date + timedelta(minutes=t) for t in times]
                
                # Create horizontal bar plot
                plt.barh(names, [60] * len(names), left=datetime_times, color=colors, alpha=0.7)
                
                # Customize plot
                plt.title(f"Day {day_idx + 1} Itinerary")
                plt.xlabel("Time")
                plt.ylabel("Locations")
                
                # Format x-axis
                plt.gca().xaxis.set_major_locator(HourLocator())
                plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
                plt.gcf().autofmt_xdate()
                
                # Add grid for readability
                plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Determine output filename
            if filename is None:
                os.makedirs("results/visualizations", exist_ok=True)
                filename = f"results/visualizations/itinerary_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Save the plot
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Timeline visualization saved to {filename}")
            return filename
        
        except Exception as e:
            logger.error(f"Error creating timeline visualization: {e}")
            return None
    
    @staticmethod
    def create_cost_breakdown_pie(problem, solution, filename=None):
        """
        Create a pie chart showing cost breakdown
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector
            filename: Optional output filename
        
        Returns:
            str: Path to the generated visualization
        """
        try:
            # Evaluate solution to get cost details
            evaluation = problem.evaluate_solution(solution)
            
            # Reshape solution into x_var
            x_var = solution[:problem.x_shape].reshape(problem.NUM_DAYS, problem.num_transport_types, 
                                                    problem.num_locations, problem.num_locations)
            
            # Initialize cost categories
            cost_categories = {
                "Hotel": problem.NUM_DAYS * problem.HOTEL_COST,
                "Public Transport": 0,
                "Taxi/Ride-sharing": 0,
                "Attractions": 0,
                "Food": 0
            }
            
            # Calculate transport costs
            for day in range(problem.NUM_DAYS):
                for j, transport_type in enumerate(problem.transport_types):
                    for k in range(problem.num_locations):
                        for l in range(problem.num_locations):
                            if k == l:
                                continue
                            
                            if x_var[day, j, k, l] == 1:
                                try:
                                    # Reshape u_var for time tracking
                                    u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
                                    
                                    transport_hour = problem.get_transport_hour(u_var[day, k])
                                    transport_data = problem.transport_matrix[
                                        (problem.locations[k]["name"], 
                                         problem.locations[l]["name"], 
                                         transport_hour)][problem.transport_types[j]]
                                    
                                    # Categorize transport costs
                                    if transport_type == "transit":
                                        cost_categories["Public Transport"] += transport_data["price"]
                                    else:
                                        cost_categories["Taxi/Ride-sharing"] += transport_data["price"]
                                    
                                    # Add attraction and hawker costs
                                    if problem.locations[l]["type"] == "attraction":
                                        cost_categories["Attractions"] += problem.locations[l]["entrance_fee"]
                                    elif problem.locations[l]["type"] == "hawker":
                                        cost_categories["Food"] += 10  # Assumed meal cost
                                except KeyError:
                                    pass
            
            # Prepare data for pie chart
            labels = list(cost_categories.keys())
            sizes = list(cost_categories.values())
            
            # Remove zero-value categories
            labels = [l for l, s in zip(labels, sizes) if s > 0]
            sizes = [s for s in sizes if s > 0]
            
            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title("Trip Cost Breakdown")
            
            # Determine output filename
            if filename is None:
                os.makedirs("results/visualizations", exist_ok=True)
                filename = f"results/visualizations/cost_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Save the plot
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Cost breakdown visualization saved to {filename}")
            return filename
        
        except Exception as e:
            logger.error(f"Error creating cost breakdown visualization: {e}")
            return None
    
    @staticmethod
    def create_satisfaction_bar_chart(problem, solution, filename=None):
        """
        Create a bar chart showing satisfaction for visited attractions and hawkers
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector
            filename: Optional output filename
        
        Returns:
            str: Path to the generated visualization
        """
        try:
            # Evaluate solution to get route details
            evaluation = problem.evaluate_solution(solution)
            
            # Reshape solution for time tracking
            u_var = solution[problem.x_shape:].reshape(problem.NUM_DAYS, problem.num_locations)
            
            # Prepare lists to store visualization data
            names = []
            satisfactions = []
            types = []
            
            # Process each daily route
            for daily_route in evaluation.get('daily_routes', []):
                for step in daily_route:
                    if step['type'] in ['attraction', 'hawker']:
                        names.append(step['name'])
                        
                        # Get satisfaction based on location type
                        if step['type'] == 'attraction':
                            satisfaction = problem.locations[step['location']].get('satisfaction', 0)
                            max_satisfaction = 10
                        else:  # hawker
                            satisfaction = problem.locations[step['location']].get('rating', 0)
                            max_satisfaction = 5
                        
                        satisfactions.append(satisfaction)
                        types.append(step['type'])
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            
            # Color mapping
            colors = ['#4CAF50' if t == 'attraction' else '#FF9800' for t in types]
            
            # Create bars
            bars = plt.bar(names, satisfactions, color=colors)
            
            # Customize plot
            plt.title("Satisfaction Ratings of Visited Locations")
            plt.xlabel("Location Name")
            plt.ylabel("Satisfaction Rating")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}', 
                         ha='center', va='bottom')
            
            # Determine output filename
            if filename is None:
                os.makedirs("results/visualizations", exist_ok=True)
                filename = f"results/visualizations/satisfaction_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Save the plot
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Satisfaction bar chart saved to {filename}")
            return filename
        
        except Exception as e:
            logger.error(f"Error creating satisfaction bar chart: {e}")
            return None
    
    @staticmethod
    def generate_comprehensive_visualization(problem, solution, output_dir=None):
        """
        Generate a comprehensive set of visualizations
        
        Args:
            problem: TravelItineraryProblem instance
            solution: Solution vector
            output_dir: Optional output directory
        
        Returns:
            dict: Paths to generated visualization files
        """
        try:
            # Create output directory if not specified
            if output_dir is None:
                output_dir = f"results/visualizations/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate visualizations
            visualizations = {
                "timeline": SolutionVisualizer.create_timeline_plot(
                    problem, solution, 
                    filename=os.path.join(output_dir, "timeline.png")
                ),
                "cost_breakdown": SolutionVisualizer.create_cost_breakdown_pie(
                    problem, solution, 
                    filename=os.path.join(output_dir, "cost_breakdown.png")
                ),
                "satisfaction_chart": SolutionVisualizer.create_satisfaction_bar_chart(
                    problem, solution, 
                    filename=os.path.join(output_dir, "satisfaction_chart.png")
                )
            }
            
            logger.info(f"Comprehensive visualizations saved to {output_dir}")
            return visualizations
        
        except Exception as e:
            logger.error(f"Error generating comprehensive visualizations: {e}")
            return {}