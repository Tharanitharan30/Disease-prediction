import { useMemo, useState } from 'react'
import './App.css'

const menuItems = ['My Dashboard', 'Prediction', 'Reports', 'Analytics', 'Patients', 'Settings']

const upcomingList = [
	{ name: 'Radhika George', time: '09:00', type: 'MRI Follow-up' },
	{ name: 'Mohan Kumar', time: '10:15', type: 'Neurology Review' },
	{ name: 'Suresh Sagar', time: '11:30', type: 'First Consultation' },
]

const surgeriesToday = [
	{ room: 'Operation Room A', patient: 'Unni Venkatesh', time: '2:00' },
	{ room: 'Operation Room B', patient: 'Gopinath', time: '3:30' },
	{ room: 'Operation Room C', patient: 'Tamizhselvi', time: '5:45' },
]

function App() {
	const [selectedFile, setSelectedFile] = useState(null)
	const [previewUrl, setPreviewUrl] = useState('')
	const [loading, setLoading] = useState(false)
	const [error, setError] = useState('')
	const [result, setResult] = useState(null)
	const [history, setHistory] = useState([])

	const today = useMemo(() => {
		return new Date().toLocaleDateString(undefined, {
			weekday: 'short',
			day: 'numeric',
			month: 'short',
			year: 'numeric',
		})
	}, [])

	const handleFileChange = (event) => {
		const file = event.target.files?.[0] || null
		setSelectedFile(file)
		setError('')
		setResult(null)

		if (!file) {
			setPreviewUrl('')
			return
		}

		const objectUrl = URL.createObjectURL(file)
		setPreviewUrl(objectUrl)
	}

	const handleSubmit = async (event) => {
		event.preventDefault()

		if (!selectedFile) {
			setError('Please choose a scan image first.')
			return
		}

		setLoading(true)
		setError('')

		try {
			const formData = new FormData()
			formData.append('image', selectedFile)

			const response = await fetch('/api/predict/brain/', {
				method: 'POST',
				body: formData,
			})

			const rawText = await response.text()
			let payload = {}

			if (rawText) {
				try {
					payload = JSON.parse(rawText)
				} catch {
					throw new Error('Server returned an unreadable response.')
				}
			}

			if (!response.ok) {
				const backendMessage = payload?.detail || payload?.error
				throw new Error(backendMessage || 'Prediction request failed.')
			}

			const finalResult = {
				file: payload.file_name || selectedFile.name,
				organ: payload.organ || 'brain',
				model: payload.model_name || payload.model || 'Brain MRI Model',
				prediction: payload.prediction || 'Unknown',
				confidence: Number(payload.confidence ?? 0),
				risk: payload.risk_level || payload.risk || 'Unknown',
				createdAt: new Date().toLocaleTimeString(),
			}

			setResult(finalResult)
			setHistory((previous) => [finalResult, ...previous].slice(0, 5))
		} catch (submitError) {
			setError(submitError.message || 'Prediction failed.')
		} finally {
			setLoading(false)
		}
	}

	return (
		<div className="scene">
			<div className="dashboard-shell">
				<aside className="sidebar">
					<div className="profile-card">
						<div className="avatar">AT</div>
						<h3>Alberto Ismail</h3>
						<p>Radiology Department</p>
					</div>

					<nav className="menu">
						{menuItems.map((item, index) => (
							<button
								key={item}
								type="button"
								className={`menu-item ${index === 0 ? 'active' : ''}`}
							>
								{item}
							</button>
						))}
					</nav>
				</aside>

				<main className="main-content">
					<header className="topbar">
						<div>
							<h1>Good Morning, Alberto!</h1>
							<p>Medical analytics and prediction center</p>
						</div>
						<div className="topbar-actions">
							<input type="search" placeholder="Search" aria-label="Search" />
							<div className="date-pill">Today {today}</div>
						</div>
					</header>

					<section className="hero-grid">
						<article className="card trend-card">
							<div className="trend-head">
								<h2>Prediction Overview</h2>
								<div className="legend">
									<span className="dot blue" />
									Tumor
									<span className="dot mint" />
									Normal
								</div>
							</div>
							<div className="sparkline">
								<div className="line" />
								<div className="line secondary" />
							</div>
							<div className="stat-row">
								<div>
									<span>Scans today</span>
									<strong>{history.length}</strong>
								</div>
								<div>
									<span>Last prediction</span>
									<strong>{result?.prediction || 'Pending'}</strong>
								</div>
								<div>
									<span>Risk level</span>
									<strong>{result?.risk || 'Unknown'}</strong>
								</div>
							</div>
						</article>

						<article className="side-widgets">
							<div className="widget-card">
								<h4>Confidence</h4>
								<strong>{result ? `${result.confidence.toFixed(2)}%` : '--'}</strong>
							</div>
							<div className="widget-card">
								<h4>Model</h4>
								<strong>{result?.model || 'Brain MRI Model'}</strong>
							</div>
							<div className="widget-card">
								<h4>Organ</h4>
								<strong>{result?.organ || 'brain'}</strong>
							</div>
						</article>
					</section>

					<section className="action-grid">
						<article className="card upload-card">
							<h2>Upload Scan</h2>
							<p>Select one brain MRI image and run prediction.</p>

							<form onSubmit={handleSubmit} className="upload-form">
								<label className="upload-control" htmlFor="scan-file">
									<input
										id="scan-file"
										type="file"
										accept="image/*"
										onChange={handleFileChange}
									/>
									<span>{selectedFile ? selectedFile.name : 'Choose image file'}</span>
								</label>

								<button type="submit" className="submit-btn" disabled={loading}>
									{loading ? 'Analyzing...' : 'Run Prediction'}
								</button>
							</form>

							{error ? <p className="error-text">{error}</p> : null}

							{previewUrl ? (
								<div className="preview-wrap">
									<img src={previewUrl} alt="Selected brain scan preview" />
								</div>
							) : null}
						</article>

						<article className="card result-card">
							<h2>Prediction Result</h2>
							{!result ? (
								<p className="muted">Result will appear after prediction.</p>
							) : (
								<dl className="result-list">
									<div>
										<dt>Prediction</dt>
										<dd>{result.prediction}</dd>
									</div>
									<div>
										<dt>Risk</dt>
										<dd>{result.risk}</dd>
									</div>
									<div>
										<dt>Confidence</dt>
										<dd>{result.confidence.toFixed(2)}%</dd>
									</div>
									<div>
										<dt>File</dt>
										<dd>{result.file}</dd>
									</div>
								</dl>
							)}
						</article>
					</section>

					<section className="tables-grid">
						<article className="card history-card">
							<h2>Upcoming Appointments</h2>
							<table>
								<thead>
									<tr>
										<th>Patient</th>
										<th>Time</th>
										<th>Type</th>
									</tr>
								</thead>
								<tbody>
									{upcomingList.map((item) => (
										<tr key={item.name}>
											<td>{item.name}</td>
											<td>{item.time}</td>
											<td>{item.type}</td>
										</tr>
									))}
								</tbody>
							</table>
						</article>

						<article className="card history-card">
							<h2>Recent Predictions</h2>
							{history.length === 0 ? (
								<p className="muted">No predictions yet.</p>
							) : (
								<table>
									<thead>
										<tr>
											<th>Time</th>
											<th>Prediction</th>
											<th>Confidence</th>
										</tr>
									</thead>
									<tbody>
										{history.map((item, index) => (
											<tr key={`${item.file}-${index}`}>
												<td>{item.createdAt}</td>
												<td>{item.prediction}</td>
												<td>{item.confidence.toFixed(2)}%</td>
											</tr>
										))}
									</tbody>
								</table>
							)}
						</article>

						<article className="card history-card">
							<h2>Surgeries Today</h2>
							<table>
								<thead>
									<tr>
										<th>Room</th>
										<th>Patient</th>
										<th>Time</th>
									</tr>
								</thead>
								<tbody>
									{surgeriesToday.map((item) => (
										<tr key={item.room}>
											<td>{item.room}</td>
											<td>{item.patient}</td>
											<td>{item.time}</td>
										</tr>
									))}
								</tbody>
							</table>
						</article>
					</section>
				</main>
			</div>
		</div>
	)
}

export default App
