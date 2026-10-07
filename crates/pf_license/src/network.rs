//! Network reachability check

use std::net::TcpStream;
use std::time::Duration;

/// Network check result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkCheckResult {
    Reachable,
    Unreachable,
}

/// Check if network is reachable by trying to connect to a well-known server
pub fn check_network_reachable() -> NetworkCheckResult {
    // Try to connect to a reliable public server (Google DNS)
    let addresses = ["dns.google:443", "1.1.1.1:53", "8.8.8.8:53"];

    for addr in &addresses {
        if let Ok(_) = TcpStream::connect_timeout(
            &addr.parse().unwrap_or_else(|_| "8.8.8.8:53".parse().unwrap()),
            Duration::from_secs(3),
        ) {
            return NetworkCheckResult::Reachable;
        }
    }

    NetworkCheckResult::Unreachable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_check() {
        let result = check_network_reachable();
        // This test may fail in offline environments
        println!("Network check result: {:?}", result);
    }
}
