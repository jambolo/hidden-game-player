#[cfg(feature = "analysis_game_tree")]
use std::cmp;
#[cfg(all(feature = "analysis_game_tree", feature = "analysis_game_state"))]
use crate::game_state::GameState;
#[cfg(feature = "analysis_game_tree")]
use serde_json::Value as Json;

use std::sync::Arc;
use std::cell::RefCell;
use std::mem;

use crate::game_state::GameState;
use crate::static_evaluator::StaticEvaluator;
use crate::transposition_table::TranspositionTable;

static SEF_QUALITY: i32 = 0;

/// A game tree search implementation using min-max strategy, alpha-beta pruning, and a transposition table.
pub struct GameTree {
    max_depth: i32,           // How deep to search
    transposition_table: Arc<dyn TranspositionTable>, // Transposition table (persistent)
    static_evaluator: Arc<dyn StaticEvaluator>,    // Static evaluator (persistent)
    response_generator: ResponseGenerator,

    #[cfg(feature = "analysis_game_tree")]
    /// Analysis data for the last move
    pub analysis_data: RefCell<AnalysisData>,
}

/// Response generator function object type.
/// 
/// # Arguments
/// * `state` - state to respond to
/// * `depth` - current ply
/// 
/// # Returns
/// list of all possible responses
/// 
/// # Note
/// The caller gains ownership of the returned states.
/// 
/// # Note
/// Returning no responses simply indicates that neither player can continue. It does not indicate the that game is
/// over or that the player has passed. If passing is allowed, then it must be included in the responses, especially
/// if is the only legal move. Similarly, if the inability to move results a loss, then the loss must be included as a
/// response.
pub type ResponseGenerator = Box<dyn Fn(&dyn GameState, i32) -> Vec<Box<dyn GameState>>>;

struct Node {
    state: Arc<dyn GameState>,
    value: f32,   // Value of the state
    quality: i32, // Quality of the value
}

type NodeList = Vec<Node>;

impl GameTree {
    /// Constructor.
    ///
    /// # Arguments
    /// * `tt` - A transposition table to be used in a search. The table is assumed to be persistent.
    /// * `sef` - The static evaluation function
    /// * `rg` - The response generator
    /// * `max_depth` - The maximum number of plies to search
    pub fn new(
        tt: Arc<dyn TranspositionTable>,
        sef: Arc<dyn StaticEvaluator>,
        rg: ResponseGenerator,
        max_depth: i32,
    ) -> Self {
        Self {
            transposition_table: tt,
            static_evaluator: sef,
            response_generator: rg,
            max_depth,
            #[cfg(feature = "analysis_game_tree")]
            analysis_data: RefCell::new(AnalysisData::new()),
        }
    }

    /// Searches for the best response to the given state.
    ///
    /// # Arguments
    /// * `s0` - The current state
    ///
    /// # Returns
    /// The chosen response is returned in s0->response_.
    pub fn find_best_response(&self, s0: &mut Arc<dyn GameState>) {
        let mut root = Node {
            state: s0.clone(),
            value: 0.0,
            quality: 0,
        };

        if s0.whose_turn() == GameState::PlayerId::ALICE {
            self.alice_search(&mut root, -f32::INFINITY, f32::INFINITY, 0);
        } else {
            self.bob_search(&mut root, -f32::INFINITY, f32::INFINITY, 0);
        }

        #[cfg(feature = "analysis_game_tree")]
        {
            self.analysis_data.borrow_mut().value = root.value;
        }
    }

    // Evaluate all of Alice's possible responses to the given state. The chosen response is the one with the highest value. The value
    // in the node is overwritten by the resulting value of the search.
    fn alice_search(&self, node: &mut Node, alpha: f32, beta: f32, depth: i32) {
        let response_depth = depth + 1;                      // Depth of responses to this state
        let quality = self.max_depth - depth;              // Quality of values at this depth (this is the depth of plies searched to
                                                        // get the results for this ply)
        let min_response_quality = self.max_depth - response_depth; // Minimum acceptable quality of responses to this state

        // Generate a list of the possible responses to this state. They are sorted in descending order hoping that a beta cutoff will
        // occur early.
        // Note: Preliminary values of the generated states are retrieved from the transposition table or computed by the static
        // evaluation function.
        let mut responses = self.generate_responses(node, depth);

        // If there are no responses, it can be assumed that the game is over and the value is the value of the current state. Return
        // without assigning a response.
        // Note: There are games in which the inability to move means that the player has lost, but that is not supported here. It must
        // be handled elsewhere or perhaps by generating and evaluating a "pass" response.
        if responses.is_empty() {
            return;
        }

        // Sort from highest to lowest
        responses.sort_by(|a, b| Self::descending_sorter(a, b));

        // Evaluate each of the responses and choose the one with the highest value
        let mut best_response = Node {
            state: Arc::new(()),
            value: -f32::INFINITY,
            quality: 0,
        };
        let mut pruned = false;
        let mut alpha = alpha;
        
        for mut response in responses {
            // If the game is not over, then let's see how Bob responds (updating the value of this response)
            if response.value < self.static_evaluator.alice_wins_value() {
                // The quality of a value is basically the depth of the search tree below it. The reason for checking the quality is
                // that some of the responses have not been fully searched. If the quality of the preliminary value is not as good as
                // the minimum quality and we haven't reached the maximum depth, then do a search. Otherwise, the response's quality is
                // as good as the quality of a search, so use the response as is.
                if response.quality < min_response_quality && response_depth < self.max_depth {
                    // Update the value of this response by searching Bob's responses to this response.
                    // Note: If no further search is possible, then the response's value and quality is already set by the static
                    // evaluation and response.state.response_ is left as nullptr.
                    self.bob_search(&mut response, alpha, beta, response_depth);
                }
            }
            #[cfg(feature = "debug_game_tree_node_info")]
            self.print_state_info(&response, depth, alpha, beta);

            // Determine if this response's value is the best so far. If so, then save the value and do alpha-beta pruning
            if response.value > best_response.value {
                // Save it
                best_response = response;

                // If Alice wins with this response, then there is no reason to look for anything better
                if best_response.value >= self.static_evaluator.alice_wins_value() {
                    break;
                }

                // alpha-beta pruning (beta cutoff) Here's how it works:
                //
                // The bob is looking for the lowest value. The 'beta' is the value of Bob's best move found so far in the previous
                // ply. If the value of this response is higher than the beta, then Bob will abandon its move leading to this response
                // because because the result is worse than the result of a move it has already found. As such, there is no reason to
                // continue.

                if best_response.value > beta {
                    // Beta cutoff
                    pruned = true;
                    #[cfg(feature = "analysis_game_tree")]
                    {
                        self.analysis_data.borrow_mut().beta_cutoffs += 1;
                    }
                    break;
                }

                // alpha-beta pruning (alpha) Here's how it works:
                //
                // Alice is looking for the highest value. The 'alpha' is the value of Alice's best move found so far. If the value of
                // this response is higher than the alpha, then obviously it is a better move for Alice. The alpha is subsequently
                // passed to Bob's search so that if it finds a response with a lower value than the alpha, it won't bother continuing
                // because it knows that Alice already has a better move and will choose it instead of allowing Bob to make a move with
                // a lower value.

                if best_response.value > alpha {
                    alpha = best_response.value;
                }
            }
        }

        // Update the value of this state
        node.value = best_response.value;
        node.quality = quality;
        node.state.set_response(Some(best_response.state));

        // Save the value of the state in the T-table if the ply was not pruned. Pruning results in an incorrect value because the
        // search was interrupted. Also, note that the value is stored only if its quality is better than the quality of the value in
        // the table.
        if !pruned {
            self.transposition_table.update(node.state.fingerprint(), node.value, node.quality);
        }

        // Note: all generated states created for this ply, except the the chosen response, are released at this point.
    }

    // Evaluate all of Bob's possible responses to the given state. The chosen response is the one with the lowest value. The value in
    // the node is overwritten by the resulting value of the search.
    fn bob_search(&self, node: &mut Node, alpha: f32, beta: f32, depth: i32) {
        let response_depth = depth + 1;                      // Depth of responses to this state
        let quality = self.max_depth - depth;              // Quality of values at this depth (this is the depth of plies searched to
                                                        // get the results for this ply)
        let min_response_quality = self.max_depth - response_depth; // Minimum acceptable quality of responses to this state

        // Generate a list of the possible responses to this state. They are sorted in ascending order hoping that a alpha cutoff will
        // occur early.
        // Note: Preliminary values of the generated states are retrieved from the transposition table or computed by the static
        // evaluation function.
        let mut responses = self.generate_responses(node, depth);

        // If there are no responses, it can be assumed that the game is over and the value is the value of the current state. Return
        // without assigning a response.
        // Note: There are games in which the inability to move means that the player has lost, but that is not supported here. It must
        // be handled elsewhere or perhaps by generating and evaluating a "pass" response.
        if responses.is_empty() {
            return;
        }

        // Sort from lowest to highest
        responses.sort_by(|a, b| Self::ascending_sorter(a, b));

        // Evaluate each of the responses and choose the one with the lowest value
        let mut best_response = Node {
            state: Arc::new(()),
            value: f32::INFINITY,
            quality: 0,
        };
        let mut pruned = false;
        let mut beta = beta;
        
        for mut response in responses {
            // If the game is not over, then let's see how Alice responds (updating the value of this response)
            if response.value > self.static_evaluator.bob_wins_value() {
                // The quality of a value is basically the depth of the search tree below it. The reason for checking the quality is
                // that some of the responses have not been fully searched. If the quality of the preliminary value is not as good as
                // the minimum quality and we haven't reached the maximum depth, then do a search. Otherwise, the response's quality is
                // as good as the quality of a search, so use the response as is.
                if response.quality < min_response_quality && response_depth < self.max_depth {
                    // Update the value of this response by searching Alice's responses to this response.
                    // Note: If no further search is possible, then the response's value and quality is already set by the static
                    // evaluation and response.state.response_ is left as nullptr.
                    self.alice_search(&mut response, alpha, beta, response_depth);
                }
            }
            #[cfg(feature = "debug_game_tree_node_info")]
            self.print_state_info(&response, depth, alpha, beta);

            // Determine if this response's value is the best so far. If so, then save the value and do alpha-beta pruning
            if response.value < best_response.value {
                // Save it
                best_response = response;

                // If Alice wins with this response, then there is no reason to look for anything better
                if best_response.value <= self.static_evaluator.bob_wins_value() {
                    break;
                }

                // alpha-beta pruning (alpha cutoff) Here's how it works:
                //
                // Alice is looking for the highest value. The 'alpha' is the value of Alice's best move found so far in the previous
                // ply. If the value of this response is lower than the alpha, then Alice will abandon its move leading to this
                // response because because the result is worse than the result of a move it has already found. As such, there is no
                // reason to continue.

                if best_response.value < alpha {
                    // Alpha cutoff
                    pruned = true;
                    #[cfg(feature = "analysis_game_tree")]
                    {
                        self.analysis_data.borrow_mut().alpha_cutoffs += 1;
                    }
                    break;
                }

                // alpha-beta pruning (beta) Here's how it works.
                //
                // Bob is looking for the lowest value. The 'beta' is the value of Bob's best move found so far. If the value of this
                // response is lower than the beta, then obviously it is a better move for Bob. The beta is subsequently passed to
                // Alice's search so that if it finds a response with a higher value than the beta, it won't bother continuing because
                // it knows that Bob already has a better move and will choose it instead of allowing Alice to make a move with a
                // higher value.

                if best_response.value < beta {
                    beta = best_response.value;
                }
            }
        }

        // Update the value of this state
        node.value = best_response.value;
        node.quality = quality;
        node.state.set_response(Some(best_response.state));

        // Save the value of the state in the T-table if the ply was not pruned. Pruning results in an incorrect value because the
        // search was interrupted. Also, note that the value is stored only if its quality is better than the quality of the value in
        // the table.
        if !pruned {
            self.transposition_table.update(node.state.fingerprint(), node.value, node.quality);
        }

        // Note: all generated states created for this ply, except the the chosen response, are released at this point.
    }

    // Generates a list of responses to the given node
    fn generate_responses(&self, node: &Node, depth: i32) -> NodeList {
        let responses = (self.response_generator)(node.state.as_ref(), depth);

        #[cfg(feature = "analysis_game_tree")]
        if depth < (AnalysisData::MAX_DEPTH as i32) {
            self.analysis_data.borrow_mut().generated_counts[depth as usize] += responses.len() as i32;
        }

        let mut rv = Vec::with_capacity(responses.len());

        // Create a list of response nodes
        for state in responses {
            let mut value = 0.0;
            let mut quality = 0;
            self.get_value(state.as_ref(), depth, &mut value, &mut quality);
            rv.push(Node {
                state: Arc::from(state),
                value,
                quality,
            });
        }

        rv
    }

    // Get the value of the state from the static evaluator or the transposition table
    fn get_value(&self, state: &dyn GameState, depth: i32, p_value: &mut f32, p_quality: &mut i32) {
        // SEF optimization:
        //
        // Since any value of any state in the T-table has already been computed by search and/or SEF, it has a quality that is at
        // least as good as the quality of the value returned by the SEF. So, if the state being evaluated is in the T-table, then the
        // value in the T-table is used instead of running the SEF because T-table lookup is so much faster than the SEF.

        // If it is in the T-table then use that value, otherwise compute the value using SEF.
        if let Some(result) = self.transposition_table.check(state.fingerprint()) {
            *p_value = result.value;
            *p_quality = result.quality;
            return;
        }

        #[cfg(feature = "analysis_game_tree")]
        if depth < (AnalysisData::MAX_DEPTH as i32) {
            self.analysis_data.borrow_mut().evaluated_counts[depth as usize] += 1;
        }

        let value = self.static_evaluator.evaluate(state);

        *p_value = value;
        *p_quality = SEF_QUALITY;

        // Save the value of the state in the T-table
        self.transposition_table.update(state.fingerprint(), *p_value, *p_quality);
    }

    #[cfg(feature = "debug_game_tree_node_info")]
    fn print_state_info(&self, node: &Node, depth: i32, alpha: f32, beta: f32) {
        for i in 0..depth {
            eprint!("{:<2}  ", i);
        }

        let fingerprint = node.state.fingerprint();
        eprint!("f = 0x{:08x}, value = {:6.2}, quality = {:3}, ", 
                fingerprint & 0xffffffff, node.value, node.quality);
        
        if alpha == -f32::INFINITY {
            eprint!("alpha = -∞, ");
        } else {
            eprint!("alpha = {:6.2}, ", alpha);
        }
        
        if beta == f32::INFINITY {
            eprintln!("beta = ∞");
        } else {
            eprintln!("beta = {:6.2}", beta);
        }
    }

    // Sort nodes in descending order by value.
    fn descending_sorter(a: &Node, b: &Node) -> std::cmp::Ordering {
        b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal)
    }

    // Sort  nodes in ascending order by value.
    fn ascending_sorter(a: &Node, b: &Node) -> std::cmp::Ordering {
        a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(feature = "analysis_game_tree")]
/// Analysis data relevant to the game tree's operation
pub struct AnalysisData {
    pub const MAX_DEPTH: usize = 10; // Maximum number of plies tracked
    pub generated_counts: [i32; Self::MAX_DEPTH],
    pub evaluated_counts: [i32; Self::MAX_DEPTH],
    pub value: f32,
    pub alpha_cutoffs: i32,
    pub beta_cutoffs: i32,
    #[cfg(feature = "analysis_game_state")]
    pub gs_analysis_data: crate::game_state::AnalysisData,
}

#[cfg(feature = "analysis_game_tree")]
impl AnalysisData {
    pub fn new() -> Self {
        Self {
            generated_counts: [0; Self::MAX_DEPTH],
            evaluated_counts: [0; Self::MAX_DEPTH],
            value: 0.0,
            alpha_cutoffs: 0,
            beta_cutoffs: 0,
            #[cfg(feature = "analysis_game_state")]
            gs_analysis_data: crate::game_state::AnalysisData::new(),
        }
    }

    pub fn reset(&mut self) {
        self.generated_counts.fill(0);
        self.evaluated_counts.fill(0);

        self.value = 0.0;
        self.alpha_cutoffs = 0;
        self.beta_cutoffs = 0;

        #[cfg(feature = "analysis_game_state")]
        self.gs_analysis_data.reset();
    }

    pub fn to_json(&self) -> Json {
        use serde_json::json;
        
        json!({
            "generatedCounts": self.generated_counts,
            "evaluatedCounts": self.evaluated_counts,
            "value": self.value,
            "alphaCutoffs": self.alpha_cutoffs,
            "betaCutoffs": self.beta_cutoffs,
            #[cfg(feature = "analysis_game_state")]
            "gameState": self.gs_analysis_data.to_json()
        })
    }
}
