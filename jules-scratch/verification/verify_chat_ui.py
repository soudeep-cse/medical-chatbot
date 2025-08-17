from playwright.sync_api import Page, expect

def test_chat_ui(page: Page):
    """
    This test verifies that the chat UI loads correctly,
    a user can send a message, and receive a response.
    """
    # 1. Arrange: Go to the application's URL.
    # The dev server should be running on http://localhost:5173
    page.goto("http://localhost:5173")

    # 2. Assert: Check that the initial UI elements are visible.
    expect(page.get_by_role("heading", name="DocBot")).to_be_visible()
    expect(page.get_by_role("button", name="+ New Chat")).to_be_visible()
    expect(page.get_by_role("heading", name="Your Personal Medical Assistant")).to_be_visible()
    expect(page.get_by_placeholder("Type your symptoms or question here...")).to_be_visible()

    # 3. Act: Type a message and send it.
    page.get_by_placeholder("Type your symptoms or question here...").fill("Hello")
    page.get_by_role("button", name="Send").click()

    # 4. Assert: Check that the user's message appears in the chat.
    expect(page.get_by_text("Hello")).to_be_visible()

    # 5. Assert: Check that the "thinking" indicator appears.
    expect(page.get_by_text("●●● Doctor is thinking…")).to_be_visible()

    # 6. Assert: Wait for the response and check that it appears.
    # This might need to be adjusted based on the actual response time.
    expect(page.get_by_text("I couldn't find an exact answer in the book. Please consult a doctor.")).to_be_visible(timeout=10000)

    # 7. Screenshot: Capture the final result for visual verification.
    page.screenshot(path="jules-scratch/verification/verification.png")
