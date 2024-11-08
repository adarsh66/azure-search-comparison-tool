// app/frontend/src/pages/DataUpload/DataUpload.tsx
import React, { useState, useCallback , useMemo} from "react";
import { Checkbox, DefaultButton, MessageBar, MessageBarType, Spinner } from "@fluentui/react";
import styles from "./DataUpload.module.css";
import axios from "axios";

interface Database {
    key: string;
    title: string;
}

const DataUpload: React.FC = () => {
    const [selectedDatabases, setSelectedDatabases] = useState<string[]>([]);
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [successMessage, setSuccessMessage] = useState<string>("");

    const databases: Database[] = useMemo(
        () => [
            { key: "aisearch", title: "AI Search" },
            { key: "mongovc", title: "CosmosDB (Mongo vCore)" },
            { key: "mongopgsql", title: "CosmosDB (pgsql)" },
            { key: "pinecone", title: "Pinecone" }
        ],
        []
    );

    const handleDatabaseChange = useCallback((_ev?: React.FormEvent<HTMLInputElement | HTMLElement>, checked?: boolean, db?: Database) => {
        if (db && db.key !== undefined) { // Ensure db is not undefined
            if (checked) {
                setSelectedDatabases(prev => [...prev, db.key]); // db is guaranteed to be a string here
            } else {
                setSelectedDatabases(prev => prev.filter(item => item !== db.key)); // Filter out the unchecked database
            }
        }
    }, []);

    const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;
        if (files && files.length > 0) {
            setFile(files[0]);
        }
    }, []);

    const handleSubmit = useCallback(async () => {
        if (!file) {
            alert("Please select a file to upload.");
            return;
        }
        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("databases", JSON.stringify(selectedDatabases));

        try {
            await axios.post("/createNewIndex", formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setSuccessMessage("Data uploaded successfully!");
        } catch (error) {
            alert("Failed to upload data.");
        } finally {
            setLoading(false);
        }
    }, [file, selectedDatabases]);

    return (
        <div className={styles.container}>
            <h1>Welcome to the Vector Search Comparison Tool</h1>
            <p>This tool allows you to compare different vector search services. The default search service uses the Wikipedia database. You can also upload your own dataset.</p>
            <input type="file" accept=".json" onChange={handleFileChange} />
            {databases.map(db => (
                <div key={db.key}>
                    <Checkbox 
                        label={db.title} 
                        checked={selectedDatabases.includes(db.key)} 
                        onChange={(_ev, checked) => handleDatabaseChange(_ev, checked, db)}
                    />
                </div>
            ))}
            <DefaultButton text="Submit" onClick={handleSubmit} disabled={loading} />
            {loading && <Spinner label="Processing..." />}
            {successMessage && <MessageBar messageBarType={MessageBarType.success}>{successMessage}</MessageBar>}
        </div>
    );
};

export default DataUpload;