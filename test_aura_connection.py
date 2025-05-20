import os
import dotenv
from neo4j import GraphDatabase, __version__ as neo4j_driver_version, DEFAULT_DATABASE
from neo4j.exceptions import ServiceUnavailable, AuthError
import logging # Import logging
import certifi

# --- Explicitly configure logging for Neo4j ---
# Get the root logger for the neo4j library
neo4j_logger = logging.getLogger("neo4j")
neo4j_logger.setLevel(logging.DEBUG)  # Set to DEBUG for most verbose output

# Create a handler (e.g., StreamHandler to print to console)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
if not neo4j_logger.hasHandlers(): # Avoid adding multiple handlers if script is re-run in same session
    neo4j_logger.addHandler(handler)
# --- End of explicit logging configuration ---


# Load environment variables from .env file
dotenv.load_dotenv()

# --- Attempt to set SSL_CERT_FILE programmatically using certifi ---
try:
    # import certifi # This was the line you added
    certifi_path = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi_path
    os.environ['REQUESTS_CA_BUNDLE'] = certifi_path # For other libs like requests
    neo4j_logger.info(f"Programmatically set SSL_CERT_FILE to: {certifi_path}")
except ImportError:
    neo4j_logger.warning("certifi package not found. Cannot programmatically set SSL_CERT_FILE.")
except Exception as e:
    neo4j_logger.error(f"Error setting SSL_CERT_FILE from certifi: {e}")
# --- End of SSL_CERT_FILE setting ---

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Strip any potential leading/trailing whitespace from loaded env vars
if NEO4J_URI:
    NEO4J_URI = NEO4J_URI.strip()
if NEO4J_USER:
    NEO4J_USER = NEO4J_USER.strip()
# Password stripping might be problematic if it intentionally contains spaces, but usually not.
# if NEO4J_PASSWORD:
#     NEO4J_PASSWORD = NEO4J_PASSWORD.strip()

print(f"Neo4j Driver Version: {neo4j_driver_version}")
print(f"Attempting to connect to Neo4j AuraDB at: '{NEO4J_URI}'") # Added quotes for visibility
print(f"Using User: '{NEO4J_USER}'") # Added quotes for visibility
# For security, avoid printing the password in logs, but confirm it's loaded
print(f"Password loaded: {'Yes' if NEO4J_PASSWORD else 'No'}")


if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    print("\nERROR: Missing one or more Neo4j connection details in your .env file (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).")
else:
    driver = None
    try:
        # Attempt to create a driver instance
        # For neo4j+s:// (Aura), you typically don't need to specify encrypted=True or trust settings,
        # as these are implied by the scheme. We can add a connection timeout.
        neo4j_logger.info(f"Attempting to create driver for URI: {NEO4J_URI}")
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=30  # seconds, default is usually 5 or 10
        )
        neo4j_logger.info("Driver object created.")
        
        print("\nDriver created. Verifying connectivity...")
        # Verify connectivity using the recommended method
        # Specifying the default database explicitly, though usually not needed for Aura's 'neo4j' db.
        # If you were using a different named database in Aura, you'd specify it here.
        neo4j_logger.info("Calling driver.verify_connectivity()")
        driver.verify_connectivity(database=DEFAULT_DATABASE) # DEFAULT_DATABASE is 'neo4j'
        neo4j_logger.info("driver.verify_connectivity() successful.")
        print("\nSUCCESS: Successfully connected to Neo4j AuraDB and verified connectivity!")

        # Optional: You can also try a simple query if verify_connectivity passes
        # print("\nAttempting a simple query...")
        # with driver.session(database="neo4j") as session: # AuraDB default database is 'neo4j'
        #     neo4j_logger.info("Attempting to run a simple test query.")
        #     result = session.run("RETURN 1 AS one")
        #     record = result.single()
        #     if record and record["one"] == 1:
        #         neo4j_logger.info("Test query successful.")
        #         print("SUCCESS: Test query executed successfully.")
        #     else:
        #         neo4j_logger.warning("Test query did not return expected result.")
        #         print("WARNING: Connected, but test query did not return expected result.")

    except AuthError as e:
        neo4j_logger.error(f"Authentication error: {e}", exc_info=True)
        print(f"\nERROR: Authentication failed. Please check your NEO4J_USER and NEO4J_PASSWORD.")
        print(f"Details: {e}")
    except ServiceUnavailable as e:
        neo4j_logger.error(f"ServiceUnavailable error: {e}", exc_info=True)
        print(f"\nERROR: Could not connect to Neo4j @ {NEO4J_URI} or failed to verify connectivity.")
        print("This often means the driver can't establish a connection or retrieve routing information.")
        print("Troubleshooting tips:")
        print("1. Double-check your NEO4J_URI. For Aura, it should start with 'neo4j+s://'.")
        print("2. Ensure your AuraDB instance is 'Running' and not paused in the Aura console.")
        print("3. Check your internet connection and any firewalls/proxies that might block outbound connections on port 7687.")
        print(f"Details: {e}")
    except Exception as e:
        neo4j_logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nERROR: An unexpected error occurred: {e}")
        print(f"Error type: {type(e)}")
    finally:
        if driver:
            neo4j_logger.info("Closing driver connection.")
            driver.close()
            print("\nConnection closed.")
        else:
            neo4j_logger.info("Driver was not initialized, no connection to close.")
