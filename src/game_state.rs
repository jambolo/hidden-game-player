#[cfg(feature = "analysis_game_state")]
use serde_json::Value as Json;

use std::sync::Arc;

/// An abstract game state.
pub trait GameState {
    /// IDs of the players.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PlayerId {
        ALICE = 0,
        BOB = 1,
    }

    /// Returns a fingerprint for this state.
    ///
    /// # Note
    /// The fingerprint is assumed to be statistically unique.
    /// 
    /// # Note
    /// This function must be implemented.
    fn fingerprint(&self) -> u64;

    /// Returns the ID of the player that will respond to this state
    ///
    /// # Returns
    /// The ID of the player that responds to this state
    ///
    /// # Note
    /// This function must be implemented.
    fn whose_turn(&self) -> Self::PlayerId;

    /// The expected response to this state, or nullptr
    fn response(&self) -> Option<Arc<dyn GameState>>;
    fn set_response(&mut self, response: Option<Arc<dyn GameState>>);

    #[cfg(feature = "analysis_game_state")]
    /// Analysis data relevant to the game state's operation
    struct AnalysisData {
        // nothing to store here yet
    }
}

#[cfg(feature = "analysis_game_state")]
impl AnalysisData {
    fn reset(&mut self) {
        // Implementation would go here
    }

    fn to_json(&self) -> Json {
        // Implementation would go here
        Json::Null
    }
}
