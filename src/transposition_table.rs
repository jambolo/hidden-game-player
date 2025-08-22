#[cfg(feature = "analysis_transposition_table")]
use serde_json::Value as Json;

use std::mem;

/// A map of game state values referenced by the states' fingerprints.
///
/// A game state can be the result of different sequences of the same (or a different) set of moves. This technique is used to
/// cache the value of a game state regardless of the moves used to reach it, thus the name "transposition" table. The purpose of
/// the "transposition" table has been extended to become simply a cache of game state values, so it is more aptly named "game
/// state value cache" -- but the old name persists.
///
/// As a speed and memory optimization in this implementation, slots in the table are not unique to the state being stored, and a
/// value may be overwritten when a new value is added. A value is overwritten only when its "quality" is less than or equal to the
/// "quality" of the value being added.
///
/// # Note
/// The fingerprint is assumed to be random and uniformly distributed. It is assumed to never be u64::MAX.
pub struct TranspositionTable {
    table: Vec<Entry>,
    max_age: i32,

    #[cfg(feature = "analysis_transposition_table")]
    pub analysis_data: std::cell::RefCell<AnalysisData>,
}

/// Result type returned by check().
pub struct CheckResult {
    /// Value of the state
    pub value: f32,
    /// Quality of the returned value
    pub quality: i32,
}

// A note about age and quality: There are expected to be collisions in the table, so the quality is used to determine if a new
// entry should replace an existing one. Now, an entry that has not been referenced for a while will probably never be
// referenced again, so it should eventually be allowed to be replaced by a newer entry, regardless of the quality of the new
// entry.
#[derive(Clone)]
struct Entry {
    fingerprint: u64, // The state's fingerprint
    value: f32,       // The state's value
    q: i16,           // The quality of the value
    age: i16,         // The number of turns since the entry has been referenced
}

impl Entry {
    const UNUSED: u64 = u64::MAX;
    
    fn clear(&mut self) {
        self.fingerprint = Self::UNUSED;
    }
}

// Check that the size of Entry is 16 bytes. The size is not required to be 16 bytes, but 16 bytes is an optimal size.
static_assertions::assert_eq_size!(f32, [u8; 4]); // float should be 32 bits
static_assertions::assert_eq_size!(Entry, [u8; 16]); // Entry should be 16 bytes

impl TranspositionTable {
    /// Constructor
    /// 
    /// # Arguments
    /// * `size` - Number of entries in the table
    /// * `max_age` - Maximum age of entries allowed in the table
    pub fn new(size: usize, max_age: i32) -> Self {
        assert!(size > 0);
        assert!(0 < max_age && max_age < (i16::MAX as i32));

        let mut table = vec![Entry {
            fingerprint: Entry::UNUSED,
            value: 0.0,
            q: 0,
            age: 0,
        }; size];

        // Invalidate all entries in the table
        for entry in &mut table {
            entry.clear();
        }

        Self {
            table,
            max_age,
            #[cfg(feature = "analysis_transposition_table")]
            analysis_data: std::cell::RefCell::new(AnalysisData::new()),
        }
    }

    /// Returns the value and quality of a state if they are stored in the table and its quality is above the specified minimum (if
    /// specified). Otherwise, None is returned.
    ///
    /// # Arguments
    /// * `fingerprint` - Fingerprint of state to be checked for
    /// * `min_q` - Minimum quality. If less 0, it is not used. (default: -1, i.e., no minimum)
    ///
    /// # Returns
    /// optional CheckResult
    pub fn check(&self, fingerprint: u64, min_q: i32) -> Option<CheckResult> {
        assert_ne!(fingerprint, Entry::UNUSED);
        assert!(min_q < (i16::MAX as i32));

        #[cfg(feature = "analysis_transposition_table")]
        {
            self.analysis_data.borrow_mut().check_count += 1;
        }

        // Find the entry
        let entry = self.find(fingerprint);
        if entry.fingerprint != fingerprint {
            #[cfg(feature = "analysis_transposition_table")]
            if entry.fingerprint != Entry::UNUSED {
                self.analysis_data.borrow_mut().collision_count += 1;
            }
            return None; // Not found
        }

        #[cfg(feature = "analysis_transposition_table")]
        {
            self.analysis_data.borrow_mut().hit_count += 1;
        }

        // SAFETY: We know the entry exists and matches the fingerprint
        unsafe {
            let entry_ptr = entry as *const Entry as *mut Entry;
            (*entry_ptr).age = 0; // The entry was accessed so reset its age
        }

        // Check the quality
        if min_q >= 0 && (entry.q as i32) < min_q {
            return None; // Insufficient quality
        }

        Some(CheckResult {
            value: entry.value,
            quality: entry.q as i32,
        })
    }

    /// # Arguments
    /// * `fingerprint` - Fingerprint of state to be stored
    /// * `value` - Value to be stored
    /// * `quality` - Quality of the value
    pub fn update(&mut self, fingerprint: u64, value: f32, quality: i32) {
        assert_ne!(fingerprint, Entry::UNUSED);
        assert!(0 <= quality && quality < (i16::MAX as i32));

        #[cfg(feature = "analysis_transposition_table")]
        {
            self.analysis_data.borrow_mut().update_count += 1;
        }

        // Find the entry for the fingerprint
        let entry = self.find_mut(fingerprint);
        let is_unused = entry.fingerprint == Entry::UNUSED;

        // If the entry is unused or if the new quality >= the stored quality, then store the new value. Note: It is assumed to be
        // better to replace values of equal quality in order to dispose of old entries that may no longer be relevant.

        if is_unused || quality >= (entry.q as i32) {
            #[cfg(feature = "analysis_transposition_table")]
            {
                let mut analysis = self.analysis_data.borrow_mut();
                if is_unused {
                    analysis.usage += 1;
                } else if entry.fingerprint == fingerprint {
                    analysis.refreshed += 1;
                } else {
                    analysis.overwritten += 1;
                }
            }
            *entry = Entry {
                fingerprint,
                value,
                q: quality as i16,
                age: 0,
            };
        } else {
            #[cfg(feature = "analysis_transposition_table")]
            {
                self.analysis_data.borrow_mut().rejected += 1;
            }
        }
    }

    /// # Arguments
    /// * `fingerprint` - Fingerprint of state to be stored
    /// * `value` - Value to be stored
    /// * `quality` - Quality of the value
    pub fn set(&mut self, fingerprint: u64, value: f32, quality: i32) {
        assert_ne!(fingerprint, Entry::UNUSED);
        assert!(0 <= quality && quality < (i16::MAX as i32));

        #[cfg(feature = "analysis_transposition_table")]
        {
            self.analysis_data.borrow_mut().update_count += 1;
        }

        // Find the entry for the fingerprint
        let entry = self.find_mut(fingerprint);

        #[cfg(feature = "analysis_transposition_table")]
        {
            let mut analysis = self.analysis_data.borrow_mut();
            if entry.fingerprint == Entry::UNUSED {
                analysis.usage += 1;
            } else if entry.fingerprint == fingerprint {
                analysis.refreshed += 1;
            } else {
                analysis.overwritten += 1;
            }
        }

        // Store the state, value and quality
        *entry = Entry {
            fingerprint,
            value,
            q: quality as i16,
            age: 0,
        };
    }

    /// The T-table is persistent. So in order to gradually dispose of entries that are no longer relevant, entries that have not been
    /// referenced for a while are removed.
    pub fn age(&mut self) {
        for entry in &mut self.table {
            if entry.fingerprint != Entry::UNUSED {
                entry.age += 1;
                if entry.age > (self.max_age as i16) {
                    entry.fingerprint = Entry::UNUSED;
                    #[cfg(feature = "analysis_transposition_table")]
                    {
                        self.analysis_data.borrow_mut().usage -= 1;
                    }
                }
            }
        }
    }

    fn index(&self, hash: u64) -> usize {
        hash as usize % self.table.len()
    }

    fn find(&self, hash: u64) -> &Entry {
        &self.table[self.index(hash)]
    }

    fn find_mut(&mut self, hash: u64) -> &mut Entry {
        &mut self.table[self.index(hash)]
    }
}

#[cfg(feature = "analysis_transposition_table")]
// Analysis data
pub struct AnalysisData {
    pub check_count: i32,     // The number of accesses of entries in the table
    pub update_count: i32,    // The number of updates to entries in the table
    pub hit_count: i32,       // The number of times an existing entry was found in the table
    pub collision_count: i32, // The number of times a different state was found in an entry
    pub rejected: i32,        // The number of times an update was rejected
    pub overwritten: i32,     // The number of times a state's entry was overwritten by a different state
    pub refreshed: i32,       // The number of times a state's entry was updated with a newer value
    pub usage: i32,           // The number of entries in use
}

#[cfg(feature = "analysis_transposition_table")]
impl AnalysisData {
    pub fn new() -> Self {
        let mut data = Self {
            check_count: 0,
            update_count: 0,
            hit_count: 0,
            collision_count: 0,
            rejected: 0,
            overwritten: 0,
            refreshed: 0,
            usage: 0,
        };
        data.reset();
        data
    }

    pub fn reset(&mut self) {
        self.check_count = 0;
        self.update_count = 0;
        self.hit_count = 0;
        self.collision_count = 0;
        self.rejected = 0;
        self.overwritten = 0;
        self.refreshed = 0;
        // Note: usage is intentionally not reset
    }

    pub fn to_json(&self) -> Json {
        use serde_json::json;
        
        json!({
            "checkCount": self.check_count,
            "updateCount": self.update_count,
            "hitCount": self.hit_count,
            "collisionCount": self.collision_count,
            "rejected": self.rejected,
            "overwritten": self.overwritten,
            "refreshed": self.refreshed,
            "usage": self.usage
        })
    }
}
