from base_problem import TravelItineraryProblem

class DebugTravelItineraryProblem(TravelItineraryProblem):
    def debug_constraint_violations(self, x, verbose=True):
        """
        Debug constraint violations by evaluating a random or provided solution
        and reporting which constraints are violated.
        
        Args:
            x: A candidate solution to evaluate, or None to generate a random one
            verbose: Whether to print detailed information about violations
            
        Returns:
            A dictionary with statistics about constraint violations
        """
        # If no solution provided, generate a random one within bounds
        if x is None:
            x = np.random.random(self.n_var)
            x = np.minimum(np.maximum(x, self.xl), self.xu)
        
        # Evaluate the solution
        out = {}
        self._evaluate(x, out)
        
        # Initialize violation tracking
        violations = {
            "equality": [],
            "inequality": [],
            "worst_eq_violation": 0,
            "worst_ineq_violation": 0,
            "total_eq_violations": 0,
            "total_ineq_violations": 0
        }
        
        # Check equality constraints (H)
        if "H" in out and len(out["H"]) > 0:
            eq_violations = np.abs(out["H"])
            violations["worst_eq_violation"] = np.max(eq_violations)
            violations["total_eq_violations"] = np.sum(eq_violations > 0.001)
            
            # Track individual violations
            for i, val in enumerate(eq_violations):
                if val > 0.001:  # Tolerance threshold
                    violations["equality"].append((i, val))
        
        # Check inequality constraints (G)
        if "G" in out and len(out["G"]) > 0:
            ineq_violations = np.maximum(0, out["G"])
            violations["worst_ineq_violation"] = np.max(ineq_violations)
            violations["total_ineq_violations"] = np.sum(ineq_violations > 0.001)
            
            # Track individual violations
            for i, val in enumerate(ineq_violations):
                if val > 0.001:  # Tolerance threshold
                    violations["inequality"].append((i, val))
        
        # Sort violations by severity
        violations["equality"].sort(key=lambda x: x[1], reverse=True)
        violations["inequality"].sort(key=lambda x: x[1], reverse=True)
        
        # Report violations
        if verbose:
            print(f"\n===== CONSTRAINT VIOLATION ANALYSIS =====")
            print(f"Equality Constraints: {violations['total_eq_violations']} violated (out of {len(out.get('H', []))})")
            print(f"Inequality Constraints: {violations['total_ineq_violations']} violated (out of {len(out.get('G', []))})")
            
            if violations["equality"]:
                print("\nTop Equality Constraint Violations:")
                for i, (idx, val) in enumerate(violations["equality"][:10]):  # Show top 10
                    constraint_desc = self.get_constraint_description("H", idx)
                    print(f"  {i+1}. H[{idx}]: {val:.4f} - {constraint_desc}")
            
            if violations["inequality"]:
                print("\nTop Inequality Constraint Violations:")
                for i, (idx, val) in enumerate(violations["inequality"][:10]):  # Show top 10
                    constraint_desc = self.get_constraint_description("G", idx)
                    print(f"  {i+1}. G[{idx}]: {val:.4f} - {constraint_desc}")
            
            print(f"Objective values: {out.get('F', 'N/A')}")
            print("=========================================\n")
        
        return violations

    def get_constraint_description(self, type, idx):
        """
        Get a human-readable description of a constraint based on its index.
        
        Args:
            type: "G" for inequality or "H" for equality constraint
            idx: The index of the constraint
            
        Returns:
            A string describing the constraint
        """
        # This mapping needs to be customized based on your problem
        if type == "G":
            # Track position in the constraint array
            pos = 0
            
            # Attraction visit limits
            if idx < 2 * self.num_attractions:
                attraction_idx = idx // 2
                attraction_name = self.locations[attraction_idx]["name"]
                if idx % 2 == 0:
                    return f"Attraction '{attraction_name}' visited at most once as source"
                else:
                    return f"Attraction '{attraction_name}' visited at most once as destination"
            pos += 2 * self.num_attractions
            
            # Hawker visit limits per day
            if idx < pos + 2 * self.num_hawkers * self.NUM_DAYS:
                rel_idx = idx - pos
                day_idx = rel_idx // (2 * self.num_hawkers)
                hawker_rel_idx = (rel_idx % (2 * self.num_hawkers)) // 2
                
                # Find the actual hawker index in locations
                hawker_idx = -1
                hawker_count = 0
                for i, loc in enumerate(self.locations):
                    if loc["type"] == "hawker":
                        if hawker_count == hawker_rel_idx:
                            hawker_idx = i
                            break
                        hawker_count += 1
                
                hawker_name = self.locations[hawker_idx]["name"] if hawker_idx >= 0 else f"Hawker {hawker_rel_idx}"
                
                if rel_idx % 2 == 0:
                    return f"Day {day_idx+1}: Hawker '{hawker_name}' visited at most once as source"
                else:
                    return f"Day {day_idx+1}: Hawker '{hawker_name}' visited at most once as destination"
            pos += 2 * self.num_hawkers * self.NUM_DAYS
            
            # Time constraints (many of these, simplify description)
            time_constraints_count = self.NUM_DAYS * self.num_transport_types * (self.num_locations - 1) 
            time_constraints_count += self.NUM_DAYS * self.num_transport_types * (self.num_locations - 1) * (self.num_locations - 2)
            
            if idx < pos + time_constraints_count:
                return "Time constraint for route timing"
            pos += time_constraints_count
            
            # Transport type constraints
            if idx < pos + self.NUM_DAYS * self.num_locations * self.num_locations:
                rel_idx = idx - pos
                day = rel_idx // (self.num_locations * self.num_locations)
                remaining = rel_idx % (self.num_locations * self.num_locations)
                src = remaining // self.num_locations
                dest = remaining % self.num_locations
                return f"Day {day+1}: Max one transport type from {self.locations[src]['name']} to {self.locations[dest]['name']}"
            pos += self.NUM_DAYS * self.num_locations * self.num_locations
            
            # Budget constraint
            if idx == pos:
                return "Total cost within budget"
            pos += 1
            
            # Min/max visit constraints
            if idx == pos:
                return "Minimum required visits"
            elif idx == pos + 1:
                return "Maximum allowed visits"
            
            return f"Unknown inequality constraint {idx}"
            
        elif type == "H":
            # Track position in the constraint array
            pos = 0
            
            # Attraction source = destination constraint
            if idx < self.num_attractions:
                attraction_idx = -1
                attraction_count = 0
                for i, loc in enumerate(self.locations):
                    if loc["type"] == "attraction":
                        if attraction_count == idx:
                            attraction_idx = i
                            break
                        attraction_count += 1
                
                attraction_name = self.locations[attraction_idx]["name"] if attraction_idx >= 0 else f"Attraction {idx}"
                return f"Attraction '{attraction_name}' must be both source and destination if visited"
            pos += self.num_attractions
            
            # Hotel starting point constraint
            if idx < pos + self.NUM_DAYS:
                day = idx - pos
                return f"Day {day+1}: Hotel must be starting point"
            pos += self.NUM_DAYS
            
            # Flow conservation
            if idx < pos + self.NUM_DAYS * self.num_locations:
                rel_idx = idx - pos
                day = rel_idx // self.num_locations
                loc = rel_idx % self.num_locations
                return f"Day {day+1}: Flow conservation at {self.locations[loc]['name']}"
            pos += self.NUM_DAYS * self.num_locations
            
            # Return to hotel
            if idx < pos + self.NUM_DAYS:
                day = idx - pos
                return f"Day {day+1}: Must return to hotel"
            pos += self.NUM_DAYS
            
            # Exactly 2 hawker visits per day
            if idx < pos + self.NUM_DAYS:
                day = idx - pos
                return f"Day {day+1}: Exactly 2 hawker visits"
            pos += self.NUM_DAYS
            
            # Exactly 1 lunch and 1 dinner hawker visit
            if idx < pos + self.NUM_DAYS * 2:
                rel_idx = idx - pos
                day = rel_idx // 2
                meal = "lunch" if rel_idx % 2 == 0 else "dinner"
                return f"Day {day+1}: Exactly 1 hawker visit during {meal}"
            
            return f"Unknown equality constraint {idx}"
        
        return "Unknown constraint type"

    def run_feasibility_analysis(self, num_samples=10):
        """
        Run a feasibility analysis by testing multiple random solutions.
        
        Args:
            num_samples: Number of random solutions to test
            
        Returns:
            Statistics about constraint violations across all samples
        """
        print(f"\n===== RUNNING FEASIBILITY ANALYSIS WITH {num_samples} SAMPLES =====")
        
        # Track the most commonly violated constraints
        eq_violation_counts = {}
        ineq_violation_counts = {}
        
        # Track overall violation statistics
        total_eq_violations = 0
        total_ineq_violations = 0
        best_total_violations = float('inf')
        best_solution = None
        
        for i in range(num_samples):
            # Generate a random solution
            x = np.random.random(self.n_var)
            x = np.minimum(np.maximum(x, self.xl), self.xu)
            
            # Evaluate constraints (without verbose output)
            violations = self.debug_constraint_violations(x, verbose=False)
            
            # Count total violations for this sample
            total_violations = violations["total_eq_violations"] + violations["total_ineq_violations"]
            
            # Update best solution if this one has fewer violations
            if total_violations < best_total_violations:
                best_total_violations = total_violations
                best_solution = x.copy()
            
            # Update violation counts
            for idx, val in violations["equality"]:
                eq_violation_counts[idx] = eq_violation_counts.get(idx, 0) + 1
            
            for idx, val in violations["inequality"]:
                ineq_violation_counts[idx] = ineq_violation_counts.get(idx, 0) + 1
            
            total_eq_violations += violations["total_eq_violations"]
            total_ineq_violations += violations["total_ineq_violations"]
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{num_samples} samples...")
        
        # Calculate average violations
        avg_eq_violations = total_eq_violations / num_samples
        avg_ineq_violations = total_ineq_violations / num_samples
        
        # Sort violation counts
        sorted_eq_violations = sorted(eq_violation_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_ineq_violations = sorted(ineq_violation_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Print summary
        print("\n----- FEASIBILITY ANALYSIS SUMMARY -----")
        print(f"Average equality violations per sample: {avg_eq_violations:.2f}")
        print(f"Average inequality violations per sample: {avg_ineq_violations:.2f}")
        
        print("\nMost frequently violated equality constraints:")
        for i, (idx, count) in enumerate(sorted_eq_violations[:5]):
            percent = (count / num_samples) * 100
            constraint_desc = self.get_constraint_description("H", idx)
            print(f"  {i+1}. H[{idx}]: {percent:.1f}% of samples - {constraint_desc}")
        
        print("\nMost frequently violated inequality constraints:")
        for i, (idx, count) in enumerate(sorted_ineq_violations[:5]):
            percent = (count / num_samples) * 100
            constraint_desc = self.get_constraint_description("G", idx)
            print(f"  {i+1}. G[{idx}]: {percent:.1f}% of samples - {constraint_desc}")
        
        print("\n----- BEST SOLUTION ANALYSIS -----")
        if best_solution is not None:
            self.debug_constraint_violations(best_solution, verbose=True)
        
        print("========================================")
        
        return {
            "eq_violation_counts": eq_violation_counts,
            "ineq_violation_counts": ineq_violation_counts,
            "avg_eq_violations": avg_eq_violations,
            "avg_ineq_violations": avg_ineq_violations,
            "best_solution": best_solution
        }