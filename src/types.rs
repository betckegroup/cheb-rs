//! Efficient evaluation of Chebychev approximations

/// Specify Chebychev points of first or second kind
#[derive(Copy, Clone)]
pub enum Kind {
    First,
    Second,
}
