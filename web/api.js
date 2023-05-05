const apiUrl = "http://127.0.0.1:3000"

export const getRegex = async (matches, rejections) => {
    if (!matches.length > 0) {
        alert("Please Add Match Texts");
        return;
    }
    if (!rejections.length > 0) {
        alert("Please Add Rejection Texts");
        return;
    }
    let request = await fetch(`${apiUrl}/get-regex`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            matches: matches,
            rejections: rejections
        })
    })

    const response = await request.json();

    return response;
}